/*
 * Copyright 2017 - 2023  John Snow Labs
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.johnsnowlabs.ml.ai

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcess.{
  ForcedTokenLogitProcessor,
  MinLengthLogitProcessor,
  SuppressLogitProcessor
}
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcessorList
import com.johnsnowlabs.ml.onnx.OnnxSession
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWrappers
import com.johnsnowlabs.ml.onnx.TensorResources.implicits._
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.{
  EncoderDecoderWrappers => OpenvinoEncoderDecoder
}
import com.johnsnowlabs.ml.tensorflow
import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.ml.tensorflow.sign.ModelSignatureConstants.EncoderOutput
import com.johnsnowlabs.ml.tensorflow.sign.ModelSignatureManager
import com.johnsnowlabs.ml.util.LinAlg.argmax
import com.johnsnowlabs.ml.util._
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.WhisperPreprocessor
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{SpecialTokens, WhisperTokenDecoder}
import com.johnsnowlabs.nlp.{Annotation, AnnotationAudio, AnnotatorType}
import org.intel.openvino.InferRequest
import org.slf4j.LoggerFactory
import org.tensorflow.{Session, Tensor}
import org.intel.openvino.{Tensor => OpenVinoTensor}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks.{break, breakable}

/** Class representing a Whisper model. Used to call the model and generate tokens.
  *
  * @param tensorflowWrapper
  *   Tensorflow Wrapper
  * @param configProtoBytes
  *   Config ProtoBytes
  * @param signatures
  *   Signatures of the model
  * @param preprocessor
  *   Whisper preprocessor to extract features
  * @param vocabulary
  *   Vocabulary for decoding
  * @param addedSpecialTokens
  *   Added special tokens
  */
private[johnsnowlabs] class Whisper(
    val tensorflowWrapper: Option[TensorflowWrapper],
    val onnxWrappers: Option[EncoderDecoderWrappers],
    val openvinoWrapper: Option[OpenvinoEncoderDecoder],
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None,
    preprocessor: WhisperPreprocessor,
    vocabulary: Map[String, Int],
    addedSpecialTokens: Map[String, Int],
    generationConfig: GenerationConfig)
    extends Serializable {

  private val logger = LoggerFactory.getLogger(this.getClass.getName)

  private val GenerationConfig(
    bosTokenId: Int,
    _,
    eosTokenId: Int,
    vocabSize: Int,
    beginSuppressTokens,
    suppressTokenIds,
    forcedDecoderIds) =
    generationConfig

  private val vocabWithAddedTokens: Map[String, Int] = vocabulary ++ addedSpecialTokens

  def tokenInVocabulary(value: String): Boolean = vocabWithAddedTokens.contains(value)

  private val tokenizerSpecialTokens: SpecialTokens =
    SpecialTokens(
      vocabWithAddedTokens,
      startTokenId = bosTokenId,
      endTokenId = eosTokenId,
      unkTokenId = eosTokenId,
      maskTokenId = eosTokenId,
      padTokenId = eosTokenId,
      additionalTokenIds = addedSpecialTokens.values.toArray)

  private val tokenDecoder: WhisperTokenDecoder =
    new WhisperTokenDecoder(vocabWithAddedTokens, tokenizerSpecialTokens)

  private val _tfWhisperSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrappers.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else throw new IllegalArgumentException("No model engine defined.")

  private val tfTensorResources = new tensorflow.TensorResources()
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  private object TfSignatures {
    object InputOps {
      val encoderInputOp: String = _tfWhisperSignatures("input_features")

      val initDecoderInputIdsOp: String = _tfWhisperSignatures("decoder_input_ids_init")
      val initDecoderEncoderStateOp: String = _tfWhisperSignatures("encoder_state_init")

      val decoderInputIdsOp: String = _tfWhisperSignatures("decoder_input_ids")
      val decoderEncoderStateOp: String = _tfWhisperSignatures("encoder_state")
      val decoderCacheOp: String = _tfWhisperSignatures("decoder_past_key_values")
      val decoderEncoderCacheOp: String = _tfWhisperSignatures("encoder_past_key_values")
    }

    object OutputOps {
      val encoderOutputOp: String = _tfWhisperSignatures("last_hidden_state")

      val initDecoderLogitsOp: String = _tfWhisperSignatures("logits_init")
      val decoderLogitsOp: String = _tfWhisperSignatures("logits")

      val decoderStateInitOp: String = _tfWhisperSignatures("decoder_past_key_values_init")
      val encoderStateInitOp: String = _tfWhisperSignatures("encoder_past_key_values_init")

      val decoderStateOp: String = _tfWhisperSignatures("decoder_past_key_values_out")
    }
  }

  private object OnnxSignatures {
    val encoderInputKey: String = "input_features"
    val encoderOutputKey = "last_hidden_state"
    val decoderOutputKey: String = "logits"

    /** Gets all keys for cache tensors.
      *
      * @param session
      *   Loaded model
      * @return
      *   All keys belonging to the state tensors
      */
    def getStateOutputKeys(session: OrtSession): Array[String] =
      session.getOutputNames.asScala.filter(_.startsWith("present.")).toArray

  }

  private object OpenVinoSignatures {
    val encoderInputIDs: String = "input_features"
    val encoderOutput: String = "last_hidden_state"
    val decoderOutputKey: String = "logits"

    def getStateOutputKeys(ovInferRequest: InferRequest): Array[String] = {

      val result = ArrayBuffer[String]()

      breakable {
        for (i <- 0 to 1000) {
          try {
            ovInferRequest.get_tensor(s"present.$i.encoder.key")
            result.append(s"present.$i.encoder.key")
            result.append(s"present.$i.encoder.value")
            result.append(s"present.$i.decoder.key")
            result.append(s"present.$i.decoder.value")
          } catch {
            case e: Exception => break
          }
        }
      }
      result.toArray
    }

    def getStateOutputKeysDecoder(ovInferRequest: InferRequest): Array[String] = {

      val result = ArrayBuffer[String]()

      breakable {
        for (i <- 0 to 1000) {
          try {
            ovInferRequest.get_tensor(s"present.$i.decoder.key")
            result.append(s"present.$i.decoder.key")
            result.append(s"present.$i.decoder.value")
          } catch {
            case e: Exception => break
          }
        }
      }
      result.toArray
    }
  }

  private def sessionWarmup(): Unit = {
    val dummyInput = Seq(AnnotationAudio(AnnotatorType.AUDIO, Array.ofDim(1), Map.empty))
    generateFromAudio(
      dummyInput,
      batchSize = 2,
      maxOutputLength = 1,
      minOutputLength = 0,
      doSample = false,
      beamSize = 1,
      numReturnSequences = 1,
      temperature = 1.0,
      topK = 1,
      topP = 1.0,
      repetitionPenalty = 1.0,
      noRepeatNgramSize = 0,
      randomSeed = None)
  }

  sessionWarmup()

  private def getLogitProcessors(
      task: Option[String] = None,
      language: Option[String] = None,
      minLength: Int = 0) = {
    val processorList = new LogitProcessorList()

    if (beginSuppressTokens.isDefined) {
      // Assuming bos token at the front, get the last forced token
      val generationBeginIdx: Int = forcedDecoderIds match {
        case Some(forcedTokens) => 1 + forcedTokens.maxBy(_._1)._1
        case None => 1
      }

      processorList.addProcess(
        new SuppressLogitProcessor(beginSuppressTokens.get, Some(generationBeginIdx)))
    }

    if (suppressTokenIds.isDefined)
      processorList.addProcess(new SuppressLogitProcessor(suppressTokenIds.get))

    val forcedTokenLogitProcessor = {
      var totalForcedDecoderIds: Map[Int, Int] = forcedDecoderIds match {
        case Some(value) => value.toMap
        case None => Map.empty
      }

      // Assumed Language and Task tokens were checked during setter
      // Language token should be at index 1. If None, then don't force anything on that position.
      totalForcedDecoderIds =
        if (language.isDefined)
          totalForcedDecoderIds.updated(1, vocabWithAddedTokens(language.get))
        else totalForcedDecoderIds

      // Task token should be at index index 2. If None, then it should be "transcribe"
      // (by default forced by official models)
      totalForcedDecoderIds =
        if (task.isDefined) totalForcedDecoderIds.updated(2, vocabWithAddedTokens(task.get))
        else totalForcedDecoderIds

      new ForcedTokenLogitProcessor(totalForcedDecoderIds.toArray)
    }

    processorList.addProcess(forcedTokenLogitProcessor)

    if (minLength > 0)
      processorList.addProcess(new MinLengthLogitProcessor(eosTokenId, minLength, vocabSize))

    processorList

  }

  /** @param batchAudio
    *   Sequence of audio floats
    * @param batchSize
    *   Batch size
    * @param minOutputLength
    *   Minimum length of output
    * @param maxOutputLength
    *   Maximum length of output
    * @param doSample
    *   Whether to sample or not
    * @param temperature
    *   Temperature for sampling
    * @param topK
    *   Top K for sampling
    * @param topP
    *   Top P for sampling
    * @param repetitionPenalty
    *   Repetition penalty for sampling
    * @param noRepeatNgramSize
    *   No repeat ngram size for sampling
    * @param randomSeed
    *   Random seed
    * @param beamSize
    *   Beam size
    * @return
    */
  def generateFromAudio(
      batchAudio: Seq[AnnotationAudio],
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
      randomSeed: Option[Long],
      task: Option[String] = None,
      language: Option[String] = None): Seq[Annotation] = {

    if (beamSize > 1)
      logger.warn(
        "Currently the Whisper model only supports greedy search. Will default to this behavior.")

    def emptyAnnotation(annotationAudio: AnnotationAudio) = {
      new Annotation(
        annotatorType = AnnotatorType.DOCUMENT,
        begin = 0,
        end = 0,
        result = "",
        metadata = annotationAudio.metadata)
    }

    val validBatchAudio = batchAudio.zipWithIndex
      .filter { case (annotationAudio, _) =>
        annotationAudio.result.nonEmpty
      }

    if (validBatchAudio.nonEmpty) {
      val validIndices = validBatchAudio.map(_._2)

      val logitProcessors: LogitProcessorList =
        getLogitProcessors(task, language, minOutputLength)

      val featuresBatch = validBatchAudio.map { case (AnnotationAudio(_, rawFloats, _), _) =>
        preprocessor.extractFeatures(rawFloats)
      }.toArray

      val batchDecoderStartIds = Array.fill(validBatchAudio.length, 1)(bosTokenId)

      val tokenIds: Array[Array[Int]] = detectedEngine match {
        case TensorFlow.name =>
          val session =
            tensorflowWrapper.get
              .getTFSessionWithSignature(
                configProtoBytes,
                savedSignatures = signatures,
                initAllTables = false)

          val encodedBatchFeatures: Tensor =
            encode(featuresBatch, Some(session), None).asInstanceOf[Tensor]

          val (initLogits, decoderCacheTensor, decoderEncoderCacheTensor) =
            initDecoderTf(encodedBatchFeatures, batchDecoderStartIds, logitProcessors, session)

          val batchInitGenerationTokenIds: Array[Array[Int]] =
            initLogits.map { logitsArray =>
              Array(bosTokenId, argmax(logitsArray))
            }

          // Generate the tokens
          val tokenIds: Array[Array[Int]] = generateGreedyTf(
            batchInitGenerationTokenIds,
            encodedBatchFeatures,
            decoderCacheTensor,
            decoderEncoderCacheTensor,
            maxOutputLength,
            logitProcessors,
            session)

          tfTensorResources.clearTensors()
          encodedBatchFeatures.close()

          tokenIds
        case ONNX.name =>
          val (encoderSession, env) = onnxWrappers.get.encoder.getSession(onnxSessionOptions)
          val decoderSession = onnxWrappers.get.decoder.getSession(onnxSessionOptions)._1
          val decoderWithPastSession =
            onnxWrappers.get.decoderWithPast.getSession(onnxSessionOptions)._1

          val encodedBatchTensor: OnnxTensor =
            encode(featuresBatch, None, Some((encoderSession, env))).asInstanceOf[OnnxTensor]

          val (initLogits, initEncoderStates, initDecoderStates) =
            initDecoderOnnx(
              batchDecoderStartIds,
              encodedBatchTensor,
              logitProcessors,
              (decoderSession, env))

          encodedBatchTensor.close()

          val batchInitGenerationTokenIds: Array[Array[Int]] =
            initLogits.map { logitsArray =>
              Array(bosTokenId, argmax(logitsArray))
            }

          val tokenIds = generateGreedyOnnx(
            batchInitGenerationTokenIds,
            replaceStateKeys(initEncoderStates),
            replaceStateKeys(initDecoderStates),
            maxOutputLength,
            logitProcessors,
            (decoderWithPastSession, env))

          tokenIds

        case Openvino.name =>
          val encoderInferRequest =
            openvinoWrapper.get.encoder.getCompiledModel().create_infer_request()
          val decoderInferRequest =
            openvinoWrapper.get.decoder.getCompiledModel().create_infer_request()
          val decoderWithPastInferRequest =
            openvinoWrapper.get.decoderWithPast.getCompiledModel().create_infer_request()
          val encodedBatchTensor: OpenVinoTensor = encodeOv(featuresBatch, encoderInferRequest)

          val (initLogits, initEncoderStates, initDecoderStates) =
            initDecoderOv(
              batchDecoderStartIds,
              encodedBatchTensor,
              logitProcessors,
              decoderInferRequest)

          val batchInitGenerationTokenIds: Array[Array[Int]] =
            initLogits.map { logitsArray =>
              Array(bosTokenId, argmax(logitsArray))
            }

          val tokenIds = generateGreedyOv(
            batchInitGenerationTokenIds,
            initEncoderStates.map { case (key, t) =>
              (key.replace("present", "past_key_values"), t)
            },
            initDecoderStates.map { case (key, t) =>
              (key.replace("present", "past_key_values"), t)
            },
            maxOutputLength,
            logitProcessors,
            decoderWithPastInferRequest)

          tokenIds

      }

      val batchDecodedIds = validIndices.zip(decode(tokenIds)).toMap

      batchAudio.zipWithIndex.map { case (annotationAudio, index) =>
        if (batchDecodedIds.contains(index)) {
          val decodedIds = batchDecodedIds(index)
          new Annotation(
            annotatorType = AnnotatorType.DOCUMENT,
            begin = 0,
            end = decodedIds.length - 1,
            result = decodedIds,
            metadata = annotationAudio.metadata)
        } else
          emptyAnnotation(annotationAudio)
      }
    } else
      batchAudio.map { annotationAudio =>
        emptyAnnotation(annotationAudio)
      }
  }

  /** Decodes a batch of generated token ids.
    *
    * @param batchTokenIds
    *   Batch of token ids of the generated text
    * @return
    *   batch of decoded sentences
    */
  def decode(batchTokenIds: Array[Array[Int]]): Seq[String] = {
    batchTokenIds.map(s => tokenDecoder.decodeTokens(s))
  }

  /** Encodes a batch of preprocessed input audio.
    *
    * @param features
    *   Batch of Whisper features
    * @return
    *   Tensor with encoded features for each batch
    */

  def encodeOv(
      features: Array[Array[Array[Float]]],
      ovInferRequest: InferRequest): OpenVinoTensor = {
    val encoderInputTensor = new org.intel.openvino.Tensor(
      Array(features.length, features.head.length, features.head.head.length),
      features.flatten.flatten)
    ovInferRequest.set_tensor(OpenVinoSignatures.encoderInputIDs, encoderInputTensor)
    ovInferRequest.infer()

    val result = ovInferRequest.get_tensor(OpenVinoSignatures.encoderOutput)
    result
  }
  def encode(
      features: Array[Array[Array[Float]]],
      tfSession: Option[Session],
      onnxSession: Option[(OrtSession, OrtEnvironment)]): AutoCloseable = {
    detectedEngine match {
      case TensorFlow.name =>
        val runner: Session#Runner =
          tfSession.get.runner

        val featuresTensors =
          tfTensorResources.createTensor[Array[Array[Array[Float]]]](features)

        val encoderOutputs: Tensor = runner
          .feed(TfSignatures.InputOps.encoderInputOp, featuresTensors)
          .fetch(TfSignatures.OutputOps.encoderOutputOp)
          .run()
          .asScala
          .head

        encoderOutputs
      case ONNX.name =>
        val (session, env) = onnxSession.get
        val encoderInputTensor = OnnxTensor.createTensor(env, features)

        val encoderOutputs: OnnxTensor = session
          .run(Map(OnnxSignatures.encoderInputKey -> encoderInputTensor).asJava)
          .getOnnxTensor(OnnxSignatures.encoderOutputKey)
        encoderInputTensor.close()
        encoderOutputs

    }
  }

  private def initDecoderTf(
      encodedInputsTensor: Tensor,
      decoderInputIds: Array[Array[Int]],
      logitProcessors: LogitProcessorList,
      session: Session): (Array[Array[Float]], Tensor, Tensor) = {
    val decoderInputIdsTensor: Tensor =
      tfTensorResources.createTensor[Array[Array[Int]]](decoderInputIds)

    val runner = session.runner
      .feed(TfSignatures.InputOps.initDecoderInputIdsOp, decoderInputIdsTensor)
      .feed(TfSignatures.InputOps.initDecoderEncoderStateOp, encodedInputsTensor)
      .fetch(TfSignatures.OutputOps.initDecoderLogitsOp)
      .fetch(TfSignatures.OutputOps.decoderStateInitOp)
      .fetch(TfSignatures.OutputOps.encoderStateInitOp)

    val decoderOuts = runner.run().asScala
    val logitsRaw = tensorflow.TensorResources.extractFloats(decoderOuts.head)
    decoderOuts.head.close()

    val rawLogits =
      logitsRaw.grouped(vocabSize).toArray // Should result in length batch size
    val logits = logitProcessors.process(decoderInputIds, rawLogits, decoderInputIds.head.length)

    val decoderStateTensor = decoderOuts(1)
    val encoderStateTensor = decoderOuts(2)

    (logits, decoderStateTensor, encoderStateTensor)
  }

  private def greedyGenerationFinished(
      decoderIds: Seq[Array[Int]],
      eosTokenId: Int,
      maxOutputLength: Int): Boolean =
    decoderIds.map(_.last).forall(_ == eosTokenId) || decoderIds.head.length == maxOutputLength

  /** Generates a Sequence of tokens with a greedy strategy.
    *
    * The token with the highest score will always be chosen.
    *
    * @param decoderEncoderCacheTensor
    *   Tensor of encoded input for the decoder
    * @param decoderCacheTensor
    *   Tensor for encoder attention mask
    * @param initInputIds
    *   Init Input IDs from the first run of the decoder
    * @param maxOutputLength
    *   Max length of the generated sequence
    * @param logitProcessor
    *   Optional logit processors
    * @param session
    *   Tensorflow session
    * @return
    */
  private def generateGreedyTf(
      initInputIds: Array[Array[Int]],
      encoderStateTensor: Tensor,
      decoderCacheTensor: Tensor,
      decoderEncoderCacheTensor: Tensor,
      maxOutputLength: Int,
      logitProcessor: LogitProcessorList,
      session: Session): Array[Array[Int]] = {

    var generatedIds = initInputIds
    var currentCacheStateTensor = decoderCacheTensor

    while (!greedyGenerationFinished(generatedIds, eosTokenId, maxOutputLength)) {

      val (rawBatchLogits: Array[Array[Float]], updatedDecoderState: Tensor) =
        getDecoderOutputTf(
          generatedIds,
          encoderStateTensor,
          currentCacheStateTensor,
          decoderEncoderCacheTensor,
          session)

      currentCacheStateTensor.close()

      val batchLogits =
        logitProcessor.process(generatedIds, rawBatchLogits, generatedIds.head.length)
      val nextTokenIds: Array[Int] = batchLogits.map(argmax)
      currentCacheStateTensor = updatedDecoderState

      generatedIds =
        generatedIds.zip(nextTokenIds).map { case (currentIds: Array[Int], nextId: Int) =>
          currentIds ++ Array(nextId)
        }
    }
    currentCacheStateTensor.close()

    generatedIds
  }

  /** Get model output for a batch of input sequences */
  private def getDecoderOutputTf(
      decoderInputIds: Array[Array[Int]],
      encoderStateTensor: Tensor,
      decoderCacheTensor: Tensor,
      decoderEncoderCacheTensor: Tensor,
      session: Session): (Array[Array[Float]], Tensor) = {

    // Only requires the last generated token
    val lastTokens: Array[Array[Int]] =
      decoderInputIds.map { tokenIds =>
        Array(tokenIds.last)
      }

    val decoderInputIdsTensor: Tensor =
      tfTensorResources.createTensor[Array[Array[Int]]](lastTokens)

    val runner = session.runner
      .feed(TfSignatures.InputOps.decoderInputIdsOp, decoderInputIdsTensor)
      .feed(TfSignatures.InputOps.decoderEncoderStateOp, encoderStateTensor)
      .feed(TfSignatures.InputOps.decoderCacheOp, decoderCacheTensor)
      .feed(TfSignatures.InputOps.decoderEncoderCacheOp, decoderEncoderCacheTensor)
      .fetch(TfSignatures.OutputOps.decoderLogitsOp)
      .fetch(TfSignatures.OutputOps.decoderStateOp)

    val decoderOuts = runner.run().asScala
    val logitsRaw = tensorflow.TensorResources.extractFloats(decoderOuts.head)

    val nextTokenLogits =
      logitsRaw.grouped(vocabSize).toArray // Should result in length batch size

    val updatedDecoderState = decoderOuts(1)

    decoderOuts.head.close()
    (nextTokenLogits, updatedDecoderState)
  }

  private def initDecoderOv(
      decoderInputIds: Array[Array[Int]],
      encoderOutputs: OpenVinoTensor,
      logitProcessors: LogitProcessorList,
      ovInferRequest: InferRequest)
      : (Array[Array[Float]], Map[String, OpenVinoTensor], Map[String, OpenVinoTensor]) = {

    val inputIdsAsLong: Array[Array[Long]] = decoderInputIds.map(_.map(_.toLong))
    val decoderInputTensor = new org.intel.openvino.Tensor(
      Array(inputIdsAsLong.length, inputIdsAsLong.head.length),
      inputIdsAsLong.flatten)
    ovInferRequest.set_tensor("input_ids", decoderInputTensor)
    ovInferRequest.set_tensor("encoder_hidden_states", encoderOutputs)
    ovInferRequest.infer()

    val rawLogits =
      ovInferRequest
        .get_tensor(OpenVinoSignatures.decoderOutputKey)
        .data
        .grouped(vocabSize)
        .toArray

    val logits = logitProcessors.process(decoderInputIds, rawLogits, decoderInputIds.head.length)

    val stateOutputKeys: Array[String] = OpenVinoSignatures.getStateOutputKeys(ovInferRequest)

    val decoderStates = mutable.Map[String, OpenVinoTensor]()
    val encoderStates = mutable.Map[String, OpenVinoTensor]()

    for (key <- stateOutputKeys) {
      val tensor: OpenVinoTensor = ovInferRequest.get_tensor(key)
      encoderStates += (key -> tensor)

    }
    for (key <- stateOutputKeys) {
      val tensor: OpenVinoTensor = ovInferRequest.get_tensor(key)
      decoderStates += (key -> tensor)
    }

    (logits, encoderStates.toMap, decoderStates.toMap)

  }

  private def initDecoderOnnx(
      decoderInputIds: Array[Array[Int]],
      encoderOutputs: OnnxTensor,
      logitProcessors: LogitProcessorList,
      onnxSession: (OrtSession, OrtEnvironment))
      : (Array[Array[Float]], Map[String, OnnxTensor], Map[String, OnnxTensor]) = {

    val (sessionRunner, env) = onnxSession

    val inputIdsAsLong = decoderInputIds.map(_.map(_.toLong))

    val decoderInputTensor: OnnxTensor = OnnxTensor.createTensor(env, inputIdsAsLong)

    val decoderInputs =
      Map("input_ids" -> decoderInputTensor, "encoder_hidden_states" -> encoderOutputs).asJava

    val sessionOutput = sessionRunner.run(decoderInputs)
    decoderInputTensor.close()

    val rawLogits =
      sessionOutput.getFloatArray(OnnxSignatures.decoderOutputKey).grouped(vocabSize).toArray

    val logits = logitProcessors.process(decoderInputIds, rawLogits, decoderInputIds.head.length)

    val stateKeys = OnnxSignatures.getStateOutputKeys(sessionRunner)
    val encoderStates =
      sessionOutput.getOnnxTensors(stateKeys)

    val decoderStates =
      sessionOutput.getOnnxTensors(stateKeys)

    (logits, encoderStates, decoderStates)
  }

  private def getDecoderOutputOnnx(
      decoderInputIds: Array[Array[Int]],
      pastEncoderStateTensors: Map[String, OnnxTensor],
      pastDecoderStateTensors: Map[String, OnnxTensor],
      onnxSession: (OrtSession, OrtEnvironment))
      : (Array[Array[Float]], Map[String, OnnxTensor]) = {

    val (session, env) = onnxSession

    // Only requires the last generated token as Long
    val lastTokens: Array[Array[Long]] =
      decoderInputIds.map { tokenIds =>
        Array(tokenIds.last.toLong)
      }

    val lastTokensTensor: OnnxTensor =
      OnnxTensor.createTensor(env, lastTokens)

    val decoderWithPastInputs: java.util.Map[String, OnnxTensor] =
      (Map(
        "input_ids" -> lastTokensTensor) ++ pastEncoderStateTensors ++ pastDecoderStateTensors).asJava

    val sessionOutput = session.run(decoderWithPastInputs)
    val logits = sessionOutput.getFloatArray(OnnxSignatures.decoderOutputKey)

    val updatedDecoderStates =
      sessionOutput.getOnnxTensors(OnnxSignatures.getStateOutputKeys(session))

    lastTokensTensor.close()

    val batchLogits = logits.grouped(vocabSize).toArray
    (batchLogits, updatedDecoderStates)
  }

  private def getDecoderOutputOv(
      decoderInputIds: Array[Array[Int]],
      pastEncoderStateTensors: Map[String, OpenVinoTensor],
      pastDecoderStateTensors: Map[String, OpenVinoTensor],
      ovInferRequest: InferRequest): (Array[Array[Float]], Map[String, OpenVinoTensor]) = {

    // Only requires the last generated token as Long
    val lastTokens: Array[Array[Long]] =
      decoderInputIds.map { tokenIds =>
        Array(tokenIds.last.toLong)
      }

    val lastTokensTensor: OpenVinoTensor =
      new org.intel.openvino.Tensor(Array(1, 1), lastTokens.flatten)

    ovInferRequest.set_tensor("input_ids", lastTokensTensor)
    pastEncoderStateTensors.foreach { case (key, value) =>
      ovInferRequest.set_tensor(key, value)
    }

    pastDecoderStateTensors.foreach { case (key, value) =>
      ovInferRequest.set_tensor(key, value)
    }
    ovInferRequest.infer()

    val logits = ovInferRequest.get_tensor(OpenVinoSignatures.decoderOutputKey)

    val stateOutputKeys: Array[String] =
      OpenVinoSignatures.getStateOutputKeysDecoder(ovInferRequest)

    val updatedDecoderStates = mutable.Map[String, OpenVinoTensor]()

    for (key <- stateOutputKeys) {
      val tensor: OpenVinoTensor = ovInferRequest.get_tensor(key)
      updatedDecoderStates += (key -> tensor)
    }

    val batchLogits = logits.data().grouped(vocabSize).toArray

    (batchLogits, updatedDecoderStates.toMap)
  }

  /** Generates Tokens in a greedy fashion.
    *
    * @param initInputIds
    *   Last Input ids for each batch after initializing the decoder
    * @param encoderStateTensors
    *   Map of each encoder state name and its tensor
    * @param decoderStateTensors
    *   Map of each decoder state name and its tensor
    * @param maxOutputLength
    *   Max output length
    * @return
    */
  private def generateGreedyOnnx(
      initInputIds: Array[Array[Int]],
      encoderStateTensors: Map[String, OnnxTensor],
      decoderStateTensors: Map[String, OnnxTensor],
      maxOutputLength: Int,
      logitProcessor: LogitProcessorList,
      onnxSession: (OrtSession, OrtEnvironment)): Array[Array[Int]] = {

    var generatedIds: Array[Array[Int]] = initInputIds
    var currentDecoderStateTensors = decoderStateTensors

    while (!greedyGenerationFinished(generatedIds, eosTokenId, maxOutputLength)) {

      val (rawBatchLogits: Array[Array[Float]], updatedDecoderStates: Map[String, OnnxTensor]) =
        getDecoderOutputOnnx(
          generatedIds,
          encoderStateTensors,
          currentDecoderStateTensors,
          onnxSession)

      val batchLogits =
        logitProcessor.process(generatedIds, rawBatchLogits, generatedIds.head.length)
      val nextTokenIds: Array[Int] = batchLogits.map(argmax)
      currentDecoderStateTensors = replaceStateKeys(updatedDecoderStates)

      generatedIds =
        generatedIds.zip(nextTokenIds).map { case (currentIds: Array[Int], nextId: Int) =>
          currentIds ++ Array(nextId)
        }
    }
    currentDecoderStateTensors.foreach { case (_, tensor) =>
      tensor.close()
    }

    generatedIds
  }

  private def generateGreedyOv(
      initInputIds: Array[Array[Int]],
      encoderStateTensors: Map[String, OpenVinoTensor],
      decoderStateTensors: Map[String, OpenVinoTensor],
      maxOutputLength: Int,
      logitProcessor: LogitProcessorList,
      ovInferRequest: InferRequest): Array[Array[Int]] = {

    var generatedIds: Array[Array[Int]] = initInputIds
    var currentDecoderStateTensors = decoderStateTensors

    while (!greedyGenerationFinished(generatedIds, eosTokenId, maxOutputLength)) {

      val (
        rawBatchLogits: Array[Array[Float]],
        updatedDecoderStates: Map[String, OpenVinoTensor]) =
        getDecoderOutputOv(
          generatedIds,
          encoderStateTensors,
          currentDecoderStateTensors,
          ovInferRequest)

      val batchLogits =
        logitProcessor.process(generatedIds, rawBatchLogits, generatedIds.head.length)
      val nextTokenIds: Array[Int] = batchLogits.map(argmax)
      currentDecoderStateTensors = updatedDecoderStates
        .map { case (key, t) =>
          (key.replace("present", "past_key_values"), t)
        }
      generatedIds =
        generatedIds.zip(nextTokenIds).map { case (currentIds: Array[Int], nextId: Int) =>
          currentIds ++ Array(nextId)
        }
    }

    generatedIds
  }

  private def replaceStateKeys(outputs: Map[String, OnnxTensor]): Map[String, OnnxTensor] =
    outputs.map { case (key, t) =>
      (key.replace("present", "past_key_values"), t)
    }

}
