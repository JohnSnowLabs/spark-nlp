/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.ai

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import com.johnsnowlabs.ml.ai.util.Generation.{Generate, GenerationConfig}
import com.johnsnowlabs.ml.onnx.OnnxSession
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWithoutPastWrappers
import com.johnsnowlabs.ml.onnx.TensorResources.implicits._
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.Gpt2Tokenizer
import org.intel.openvino.InferRequest
import org.tensorflow.{Session, Tensor}

import scala.collection.JavaConverters._

private[johnsnowlabs] class VisionEncoderDecoder(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrappers: Option[EncoderDecoderWithoutPastWrappers],
    configProtoBytes: Option[Array[Byte]] = None,
    tokenizer: Gpt2Tokenizer,
    preprocessor: Preprocessor,
    generationConfig: GenerationConfig,
    signatures: Option[Map[String, String]] = None)
    extends Generate {

  val _tfSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  val tensorResources = new TensorResources()
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrappers.isDefined) ONNX.name
    else throw new IllegalArgumentException("No model engine defined.")
  private def sessionWarmup(): Unit = {
    val nChannels = 3
    val dummyInput = Array(
      AnnotationImage(
        AnnotatorType.IMAGE,
        "",
        preprocessor.size,
        preprocessor.size,
        nChannels,
        0,
        Array.fill[Byte](preprocessor.size * preprocessor.size * nChannels)(0.toByte),
        Map.empty))

    generateFromImage(
      images = dummyInput,
      batchSize = 1,
      maxOutputLength = 1,
      minOutputLength = 0,
      doSample = false,
      beamSize = 1,
      numReturnSequences = 1,
      temperature = 1.0,
      topK = 1,
      topP = 1.0,
      repetitionPenalty = 0,
      noRepeatNgramSize = 0,
      randomSeed = None)
  }

  sessionWarmup()

  private object TfSignatures {
    object InputOps {
      val encoderInput = _tfSignatures
        .getOrElse(ModelSignatureConstants.PixelValuesInput.key, "missing_pixel_values")
      val decoderEncoderState = _tfSignatures.getOrElse(
        ModelSignatureConstants.DecoderEncoderInputIds.key,
        ModelSignatureConstants.DecoderEncoderInputIds.value)
      val decoderInputIds = _tfSignatures.getOrElse(
        ModelSignatureConstants.DecoderInputIds.key,
        ModelSignatureConstants.DecoderInputIds.value)
    }

    object OutputOps {
      val encoderState = _tfSignatures
        .getOrElse(
          ModelSignatureConstants.LastHiddenState.key,
          ModelSignatureConstants.LastHiddenState.value)
      val decoderLogits = _tfSignatures
        .getOrElse(
          ModelSignatureConstants.LogitsOutput.key,
          ModelSignatureConstants.LogitsOutput.value)
    }
  }

  private object OnnxSignatures {
    val encoderInputIdsTensor: String = "pixel_values"
    val encoderOutputKey = "last_hidden_state"
    val decoderOutputKey: String = "logits"
    val decoderInputIDs: String = "input_ids"
    val decoderEncoderState: String = "encoder_hidden_states"

  }

  private def preprocessImages(
      annotations: Array[AnnotationImage]): Array[Array[Array[Array[Float]]]] = {

    val batchProcessedImages = annotations.map { annot =>
      val bufferedImage = ImageIOUtils.byteToBufferedImage(
        bytes = annot.result,
        w = annot.width,
        h = annot.height,
        nChannels = annot.nChannels)

      val resizedImage = if (preprocessor.do_resize) {
        ImageResizeUtils.resizeBufferedImage(
          width = preprocessor.size,
          height = preprocessor.size,
          preprocessor.resample)(bufferedImage)
      } else bufferedImage

      val normalizedImage =
        ImageResizeUtils.normalizeAndConvertBufferedImage(
          img = resizedImage,
          mean = preprocessor.image_mean,
          std = preprocessor.image_std,
          doNormalize = preprocessor.do_normalize,
          doRescale = preprocessor.do_rescale,
          rescaleFactor = preprocessor.rescale_factor)

      normalizedImage
    }

    batchProcessedImages
  }

  /** Encodes processed images with the encoder.
    *
    * Expands the initial encoded images to match the requested beam size.
    *
    * @param batch
    *   Batch of images with dimensions (batchSize, R, G, B)
    * @return
    *   Tensor with encoded representations of the batch
    */
  private def encodeImages(
      batch: Array[Array[Array[Array[Float]]]],
      beamSize: Int,
      tfSession: Option[Session],
      onnxSession: Option[(OrtSession, OrtEnvironment)]): AutoCloseable = {

    val batchForBeams =
      batch.flatMap(imageFloats => Array.fill(beamSize)(imageFloats))
    detectedEngine match {
      case TensorFlow.name =>
        // Expand the array for each beam

        val imageTensors = tensorResources.createTensor(batchForBeams)

        val runner = tfSession.get.runner
          .feed(TfSignatures.InputOps.encoderInput, imageTensors)
          .fetch(TfSignatures.OutputOps.encoderState)

        val outs = runner.run().asScala
        outs.head

      case ONNX.name =>
        val (session, env) = onnxSession.get
        val imageTensors = OnnxTensor.createTensor(env, batchForBeams)
        val encoderResults = session
          .run(Map(OnnxSignatures.encoderInputIdsTensor -> imageTensors).asJava)
        val output = encoderResults
          .get(OnnxSignatures.encoderOutputKey)
          .get()
          .asInstanceOf[OnnxTensor]
        output

      case _ =>
        throw new IllegalArgumentException("Unknown engine type.")
    }
  }

  def generateFromImage(
      images: Array[AnnotationImage],
      batchSize: Int,
      maxOutputLength: Int,
      minOutputLength: Int,
      doSample: Boolean,
      beamSize: Int,
      numReturnSequences: Int,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      randomSeed: Option[Long]): Seq[Annotation] = {

    val captions: Seq[Annotation] = images
      .grouped(batchSize)
      .flatMap { batch =>
        val batchSize = batch.length
        val preprocessedImages = preprocessImages(images)

        val batchDecoderStartIds = Array.fill(batchSize, 1)(generationConfig.bosId)
        val encoderIds: Array[Array[Int]] = Array.fill(batchDecoderStartIds.length)(Array.empty)
        val generatedTokenIds =
          detectedEngine match {
            case TensorFlow.name =>
              val session: Session = tensorflowWrapper.get
                .getTFSessionWithSignature(
                  configProtoBytes = configProtoBytes,
                  initAllTables = false)
              val encodedImages = encodeImages(preprocessedImages, beamSize, Some(session), None)
                .asInstanceOf[Tensor]
              generate(
                inputIds = encoderIds,
                decoderEncoderStateTensors = Left(encodedImages),
                encoderAttentionMaskTensors = null,
                decoderInputs = batchDecoderStartIds,
                maxOutputLength,
                minOutputLength,
                doSample,
                beamSize,
                numReturnSequences,
                temperature,
                topK,
                topP,
                repetitionPenalty,
                noRepeatNgramSize,
                generationConfig.vocabSize,
                generationConfig.eosId,
                generationConfig.padId,
                randomSeed,
                Array.empty,
                Left(session))
            case ONNX.name =>
              val (encoderSession, encoderEnv) =
                onnxWrappers.get.encoder.getSession(onnxSessionOptions)
              val (decoderSession, decoderEnv) =
                onnxWrappers.get.decoder.getSession(onnxSessionOptions)
              val encodedImages =
                encodeImages(
                  preprocessedImages,
                  beamSize,
                  None,
                  Some((encoderSession, encoderEnv)))
                  .asInstanceOf[OnnxTensor]
              generate(
                inputIds = batchDecoderStartIds,
                decoderEncoderStateTensors = Right(encodedImages),
                encoderAttentionMaskTensors =
                  Right(OnnxTensor.createTensor(encoderEnv, Array(1))),
                decoderInputs = batchDecoderStartIds,
                maxOutputLength,
                minOutputLength,
                doSample,
                beamSize,
                numReturnSequences,
                temperature,
                topK,
                topP,
                repetitionPenalty,
                noRepeatNgramSize,
                generationConfig.vocabSize,
                generationConfig.eosId,
                generationConfig.padId,
                randomSeed,
                Array.empty,
                Right((decoderEnv, decoderSession)))

          }

        val decodedStringsBatch = generatedTokenIds.map(tokenizer.decodeTokens).map(_.trim)

        batch.zip(decodedStringsBatch).map { case (image, caption) =>
          val imageMeta = Map(
            "height" -> image.height.toString,
            "width" -> image.width.toString,
            "nChannels" -> image.nChannels.toString,
            "mode" -> image.mode.toString,
            "origin" -> image.origin)

          Annotation(
            annotatorType = AnnotatorType.DOCUMENT,
            begin = 0,
            end = caption.length - 1,
            result = caption,
            metadata = Map("image" -> "0") ++ imageMeta)
        }
      }
      .toSeq

    // tensorResources.clearTensors()

    captions
  }

  /** Calls the model and returns the output logits.
    *
    * @param encoderInputIds
    *   Input IDs for the, not used for vision encoders
    * @param decoderInputIds
    *   Input IDs for the Decoder
    * @param decoderEncoderStateTensors
    *   Tensor of encoded pixel values for the decoder
    * @param encoderAttentionMaskTensors
    *   Tensor for encoder attention mask, not used
    * @param maxLength
    *   Max length of the input
    * @param session
    *   Tensorflow Session
    * @return
    *   Logits for the input
    */
  override def getModelOutput(
      encoderInputIds: Seq[Array[Int]],
      decoderInputIds: Seq[Array[Int]],
      decoderEncoderStateTensors: Either[Tensor, OnnxTensor],
      encoderAttentionMaskTensors: Either[Tensor, OnnxTensor],
      maxLength: Int,
      session: Either[Session, (OrtEnvironment, OrtSession)],
      ovInferRequest: Option[InferRequest]): Array[Array[Float]] = {
    getModelOutput(decoderInputIds, decoderEncoderStateTensors, session)
  }

  def getModelOutput(
      decoderInputIds: Seq[Array[Int]],
      decoderEncoderStateTensors: Either[Tensor, OnnxTensor],
      session: Either[Session, (OrtEnvironment, OrtSession)]) = {

    val decoderEncoderStateTensor = decoderEncoderStateTensors.fold(
      tfTensor => {
        // not implemented yet
        null
      },
      onnxTensor => onnxTensor)

    detectedEngine match {
      case TensorFlow.name =>
        val decoderInputIdsTensor = tensorResources.createTensor(decoderInputIds.toArray)

        val runner =
          session.left.get
            .runner()
            .feed(TfSignatures.InputOps.decoderEncoderState, decoderEncoderStateTensors.left.get)
            .feed(TfSignatures.InputOps.decoderInputIds, decoderInputIdsTensor)
            .fetch(TfSignatures.OutputOps.decoderLogits)

        val decoderOuts = runner.run().asScala
        val logitsRaw = TensorResources.extractFloats(decoderOuts.head)
        decoderOuts.head.close()

        val logits = logitsRaw.grouped(generationConfig.vocabSize)

        logits.toArray
      case ONNX.name =>
        val (env, decoderSession) = session.right.get
        val decoderInputIdsLong: Array[Array[Long]] =
          decoderInputIds.toArray.map { tokenIds => tokenIds.map(_.toLong) }

        val decoderInputIdsTensor =
          OnnxTensor.createTensor(env, decoderInputIdsLong)

        val decoderInputs: java.util.Map[String, OnnxTensor] = Map(
          OnnxSignatures.decoderInputIDs -> decoderInputIdsTensor,
          OnnxSignatures.decoderEncoderState -> decoderEncoderStateTensor).asJava
        val sessionOutput = decoderSession.run(decoderInputs)

        val sequenceLength = decoderInputIds.head.length
        val batchSize = decoderInputIds.length

        val logitsRaw = sessionOutput.getFloatArray(OnnxSignatures.decoderOutputKey)
        val decoderOutputs = (0 until batchSize).map(i => {
          logitsRaw
            .slice(
              i * sequenceLength * generationConfig.vocabSize + (sequenceLength - 1) * generationConfig.vocabSize,
              i * sequenceLength * generationConfig.vocabSize + sequenceLength * generationConfig.vocabSize)
        })
        decoderOutputs.toArray

    }
  }

}
