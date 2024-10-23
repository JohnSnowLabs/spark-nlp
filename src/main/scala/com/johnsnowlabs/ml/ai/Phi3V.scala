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

import breeze.optimize.BatchSize
import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.Phi3VWrappers
import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.tensorflow.{TensorResources, TensorflowWrapper}
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.Phi3vUtils
import org.intel.openvino.InferRequest

import scala.collection.JavaConverters._

private[johnsnowlabs] class Phi3V(
    val openvinoWrapper: Option[Phi3VWrappers],
    val onnxWrappers: Option[DecoderWrappers],
    preprocessor: Preprocessor,
    val spp: SentencePieceWrapper,
    generationConfig: GenerationConfig)
    extends Serializable {

  private def sessionWarmup(): Unit = {
    val image =
      ImageIOUtils.loadImage(getClass.getResourceAsStream("/image/ox.JPEG"))
    val bytes = ImageIOUtils.bufferedImageToByte(image.get)
    val images =
      Array(AnnotationImage("image", "ox.JPEG", 265, 360, 3, 16, bytes, Map("image" -> "0")))
//    val encoded = encode(images, preprocessor)
//    tag(encoded)
  }

  val detectedEngine: String =
    if (onnxWrappers.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else Openvino.name

  private val GenerationConfig(
    bosTokenId: Int,
    paddingTokenId: Int,
    eosTokenId: Int,
    vocabSize: Int,
    beginSuppressTokens,
    suppressTokenIds,
    forcedDecoderIds) =
    generationConfig

  private val pieceSize = spp.getSppModel.getPieceSize

  /** Decode a sequence of sentences
    * @param sentences
    *   Sequence of sentences
    * @return
    *   Sequence of decoded sentences
    */
  def decode(sentences: Array[Array[Int]]): Seq[String] = {
    sentences.map { s =>
      val filteredPieceIds = s.filter(x => x <= pieceSize)
      spp.getSppModel.decodeIds(filteredPieceIds.map(_.toInt): _*)
    }
  }
  def encode(
      imageAnnotations: Seq[AnnotationImage],
      sentences: Seq[Annotation],
      numOfCrops: Int = 17): (
      Seq[Array[Int]],
      (Array[Array[Array[Array[Array[Float]]]]], Array[Array[Int]], List[Int])) = {
    val preprocessedImages = preprocessImage(imageAnnotations, numOfCrops)
    val encodedText = encodeText(sentences).toArray

    (encodedText, preprocessedImages)
  }

  def tag(
      batch: Seq[Array[Int]],
      images: (Array[Array[Array[Array[Array[Float]]]]], Array[Array[Int]]),
      minOutputLength: Int,
      maxOutputLength: Int,
      doSample: Boolean,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      randomSeed: Option[Long],
      ignoreTokenIds: Array[Int] = Array(),
      beamSize: Int,
      maxInputLength: Int,
      stopTokenIds: Array[Int],
      numOfCrops: Int = 17): Array[Array[Int]] = {

    val (pixelValues, imageSizes) = images
    val ignoreTokenIdsInt = ignoreTokenIds
    val expandedDecoderInputsVals = batch
    val sequencesLength = expandedDecoderInputsVals.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max // - curLen
//    val pixelValues = images._1
//    val imageSizes = images._2
    val numReturn_sequences = 1
    // from config

    var effectiveBatch_size = 1
    var effectiveBatch_mult = 1

    if (doSample) {
      effectiveBatch_size = expandedDecoderInputsVals.length * numReturn_sequences
      effectiveBatch_mult = numReturn_sequences
    } else {
      effectiveBatch_size = expandedDecoderInputsVals.length
      effectiveBatch_mult = 1
    }

    val inferRequestWTE = openvinoWrapper.get.wte.getCompiledModel().create_infer_request()
    val inferRequestReshape =
      openvinoWrapper.get.reshape.getCompiledModel().create_infer_request()
    val inferRequestLanguageModel =
      openvinoWrapper.get.languageModel.getCompiledModel().create_infer_request()

    val generatedIds = generateGreedy(
      batch.toArray,
      batch.toArray,
      pixelValues,
      imageSizes,
      maxOutputLength,
      numOfCrops,
      inferRequestWTE,
      inferRequestReshape,
      inferRequestLanguageModel)
    generatedIds
  }

  def generateGreedy(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      pixelValues: Array[Array[Array[Array[Array[Float]]]]],
      imageSizes: Array[Array[Int]],
      maxOutputLength: Int,
      numOfCrops: Int,
      inferRequestWTE: InferRequest,
      inferRequestReshape: InferRequest,
      inferRequestLanguageModel: InferRequest): Array[Array[Int]] = {

    var generatedIds: Array[Array[Int]] = Array()
    var decoderInputIdsCopied = decoderInputIds
    while (!greedyGenerationFinished(
        generatedIds.map(_.map(_.toInt)),
        eosTokenId,
        maxOutputLength)) {
      val decoderOutputs = getModelOutputs(
        encoderInputIds,
        decoderInputIdsCopied,
        pixelValues,
        imageSizes,
        numOfCrops,
        inferRequestWTE,
        inferRequestReshape,
        inferRequestLanguageModel)

      val nextTokenIds = decoderOutputs.map { scores =>
        argmax(scores)
      }

      generatedIds =
        generatedIds.zip(nextTokenIds).map { case (currentIds: Array[Int], nextId: Int) =>
          currentIds ++ Array(nextId)
        }

      // extend decoder input ids
      decoderInputIdsCopied =
        decoderInputIdsCopied.zip(nextTokenIds).map { case (currentIds, nextId) =>
          currentIds ++ Array(nextId)
        }
    }
    generatedIds
  }

  def predict(
      sentences: Seq[Annotation],
      imageAnnotations: Seq[AnnotationImage],
      batchSize: Int,
      minOutputLength: Int,
      maxOutputLength: Int,
      doSample: Boolean,
      temperature: Double,
      topK: Int,
      topP: Double,
      repetitionPenalty: Double,
      noRepeatNgramSize: Int,
      randomSeed: Option[Long] = None,
      ignoreTokenIds: Array[Int] = Array(),
      beamSize: Int,
      maxInputLength: Int): Seq[Annotation] = {

    val (encodedText, preprocessedImages) = encode(imageAnnotations, sentences)
    val (pixelValues, imageSizes, imgTokens) = preprocessedImages
    val tagged = tag(
      encodedText,
      (pixelValues, imageSizes),
      minOutputLength,
      maxOutputLength,
      doSample,
      temperature,
      topK,
      topP,
      repetitionPenalty,
      noRepeatNgramSize,
      randomSeed,
      ignoreTokenIds,
      beamSize,
      maxInputLength,
      Array(eosTokenId))
    val decoded = decode(tagged)
    var sentBegin, nextSentEnd = 0
    val annotations = decoded.map { content =>
      nextSentEnd += content.length - 1
      val annots = new Annotation(
        annotatorType = DOCUMENT,
        begin = sentBegin,
        end = nextSentEnd,
        result = content,
        metadata = Map())
      sentBegin += nextSentEnd + 1
      annots
    }
    annotations
  }

  def getModelOutputs(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      pixelValues: Array[Array[Array[Array[Array[Float]]]]],
      imageSizes: Array[Array[Int]],
      numOfCrops: Int,
      inferRequestWTE: InferRequest,
      inferRequestReshape: InferRequest,
      inferRequestLanguageModel: InferRequest): Array[Array[Float]] = {

    val imageEmbeddings = getImageEmbeddings(
      encoderInputIds,
      decoderInputIds,
      pixelValues,
      imageSizes,
      numOfCrops,
      inferRequestReshape,
      inferRequestWTE)

    val (inputIdsLong, inputPositionIDsLong): (Array[Long], Array[Long]) =
      if (encoderInputIds.head.length == decoderInputIds.head.length) {
        // First pass
        val inpIdsLong = decoderInputIds.flatMap { tokenIds => tokenIds.map(_.toLong) }
        val posIdsLong = decoderInputIds.flatMap { tokenIds =>
          tokenIds.zipWithIndex.map { case (_, i) =>
            i.toLong
          }
        }
        (inpIdsLong, posIdsLong)
      } else {
        // Subsequent passes
        val inpIdsLong = decoderInputIds.map { tokenIds => tokenIds.last.toLong }
        val posIdsLong = decoderInputIds.map { tokenIds =>
          tokenIds.zipWithIndex.map { case (_, i) =>
            i.toLong
          }.last
        }
        (inpIdsLong, posIdsLong)
      }
    val attentionMask: Array[Long] =
      decoderInputIds.flatMap { tokenIds => tokenIds.map(_ => 1L) }

    val batchSize: Int = decoderInputIds.length
    val beamIdx: Array[Int] = new Array[Int](batchSize)
    val shape: Array[Int] = Array(batchSize, inputIdsLong.length / batchSize)

    val decoderAttentionMask: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize, decoderInputIds.head.length), attentionMask)
    val decoderPositionIDs: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(shape, inputPositionIDsLong)
    val beamIdxTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize), beamIdx)

    inferRequestLanguageModel.set_tensor("inputs_embeds", imageEmbeddings)
    inferRequestLanguageModel.set_tensor("attention_mask", decoderAttentionMask)
    inferRequestLanguageModel.set_tensor("position_ids", decoderPositionIDs)
    inferRequestLanguageModel.set_tensor("beam_idx", beamIdxTensor)

    inferRequestLanguageModel.infer()

    val result = inferRequestLanguageModel.get_tensor("logits")
    val logitsRaw = result.data()

    val sequenceLength = inputIdsLong.length / batchSize
    val decoderOutputs = (0 until batchSize).map(i => {
      logitsRaw
        .slice(
          i * sequenceLength * vocabSize + (sequenceLength - 1) * vocabSize,
          i * sequenceLength * vocabSize + sequenceLength * vocabSize)
    })
    decoderOutputs.toArray
  }

  def encodeText(sentences: Seq[Annotation]): Seq[Array[Int]] = {
    sentences.map(s => {
      val sentWithTask = "_" + s.result
      Array(bosTokenId) ++ spp.getSppModel.encodeAsIds(sentWithTask)
    })
  }

  private def argmax(scores: Array[Float]): Int =
    scores.zipWithIndex.maxBy { case (score, _) =>
      score
    }._2

  private def greedyGenerationFinished(
      decoderIds: Seq[Array[Int]],
      eosTokenId: Int,
      maxOutputLength: Int): Boolean =
    decoderIds.map(_.last).forall(_ == eosTokenId) || decoderIds.head.length == maxOutputLength

  def preprocessImage(imageAnnotations: Seq[AnnotationImage], numOfCrops: Int = 17)
      : (Array[Array[Array[Array[Array[Float]]]]], Array[Array[Int]], List[Int]) = {

    val hdTransformedImage = imageAnnotations
      .map(annotations => {
        val bufferedImage = ImageIOUtils.byteToBufferedImage(
          bytes = annotations.result,
          w = annotations.width,
          h = annotations.height,
          nChannels = annotations.nChannels)

        Phi3vUtils.HDTransform(bufferedImage, numOfCrops)
      })
      .toList
    val (processedImages, imageSizes, imgTokens) =
      Phi3vUtils.processHdImages(hdTransformedImage, numOfCrops)
    val pixelValues =
      Phi3vUtils.processedImagesTo5DArray(processedImages, normalize = true)
    (pixelValues, imageSizes, imgTokens)
  }

  def getImageEmbeddings(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      pixelValues: Array[Array[Array[Array[Array[Float]]]]],
      imageSizes: Array[Array[Int]],
      numOfCrops: Int,
      inferRequestReshape: InferRequest,
      inferRequestWTE: InferRequest): org.intel.openvino.Tensor = {
    val inputIdsLong: Array[Long] =
      if (encoderInputIds.head.length == decoderInputIds.head.length) {
        // First pass
        val inpIdsLong = decoderInputIds.flatMap { tokenIds => tokenIds.map(_.toLong) }

        inpIdsLong
      } else {
        // Subsequent passes
        val inpIdsLong = decoderInputIds.map { tokenIds => tokenIds.last.toLong }
        inpIdsLong
      }
    val batchSize: Int = decoderInputIds.length
    val shape: Array[Int] = Array(batchSize, inputIdsLong.length / batchSize)
    val inputIdsLongTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(shape, inputIdsLong)

    val imageEmbeddings: org.intel.openvino.Tensor =
      if (encoderInputIds.head.length == decoderInputIds.head.length) {
        val pixelValuesTensor: org.intel.openvino.Tensor =
          new org.intel.openvino.Tensor(
            Array(batchSize, numOfCrops, 3, 336, 336),
            pixelValues.flatten.flatten.flatten.flatten.map(_.toFloat))

        val imageSizesTensor: org.intel.openvino.Tensor =
          new org.intel.openvino.Tensor(Array(batchSize, 2), imageSizes.flatten.map(_.toFloat))
        inferRequestReshape.set_tensor("input_ids", inputIdsLongTensor)
        inferRequestReshape.set_tensor("pixel_values", pixelValuesTensor)
        inferRequestReshape.set_tensor("image_sizes", imageSizesTensor)

        inferRequestReshape.infer()

        inferRequestReshape.get_output_tensor()

      } else {
        inferRequestWTE.set_tensor("input_ids", inputIdsLongTensor)

        inferRequestWTE.infer()

        inferRequestReshape.get_output_tensor()
      }
    imageEmbeddings
  }

}
