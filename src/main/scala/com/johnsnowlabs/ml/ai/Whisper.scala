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
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcess.{
  ForcedTokenLogitProcessor,
  SuppressLogitProcessor
}
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcessorList
import com.johnsnowlabs.ml.ai.util.Generation.{Generate, GenerationConfig}
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWrappers
import com.johnsnowlabs.ml.onnx.TensorResources.implicits._
import com.johnsnowlabs.ml.tensorflow
import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.util._
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.WhisperPreprocessor
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{SpecialTokens, WhisperTokenDecoder}
import com.johnsnowlabs.nlp.{Annotation, AnnotationAudio, AnnotatorType}
import org.slf4j.LoggerFactory
import org.tensorflow.{Session, Tensor}

import scala.collection.JavaConverters._

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
    configProtoBytes: Option[Array[Byte]] = None,
    signatures: Option[Map[String, String]] = None,
    preprocessor: WhisperPreprocessor,
    vocabulary: Map[String, Int],
    addedSpecialTokens: Map[String, Int],
    generationConfig: GenerationConfig)
    extends Serializable
    with Generate {

  private val logger = LoggerFactory.getLogger(this.getClass.getName)

  private val GenerationConfig(
    bosTokenId: Int,
    paddingTokenId: Int,
    eosTokenId: Int,
    logitsSize: Int,
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

  val tokenDecoder: WhisperTokenDecoder =
    new WhisperTokenDecoder(vocabWithAddedTokens, tokenizerSpecialTokens)

  private val _tfWhisperSignatures: Map[String, String] =
    signatures.getOrElse(ModelSignatureManager.apply())

  val detectedEngine: String =
    if (tensorflowWrapper.isDefined) TensorFlow.name
    else if (onnxWrappers.isDefined) ONNX.name
    else throw new IllegalArgumentException("No model engine defined.")

  private val tfTensorResources = new tensorflow.TensorResources()
//  val onnxTensorResources = new onnx.TensorResources(OrtEnvironment.getEnvironment())
  private object TfSignatures {
    val encoderInputOp: String = _tfWhisperSignatures.getOrElse(
      ModelSignatureConstants.EncoderInputIds.key,
      ModelSignatureConstants.EncoderInputIds.value)
    val encoderOutputOp: String = _tfWhisperSignatures.getOrElse(
      ModelSignatureConstants.EncoderOutput.key,
      ModelSignatureConstants.EncoderOutput.value)
    val decoderEncoderOutputsOp: String = _tfWhisperSignatures.getOrElse(
      ModelSignatureConstants.DecoderEncoderInputIds.key,
      ModelSignatureConstants.DecoderEncoderInputIds.value)
    val decoderInputIdsOp: String = _tfWhisperSignatures.getOrElse(
      ModelSignatureConstants.DecoderInputIds.key,
      ModelSignatureConstants.DecoderInputIds.value)
    val decoderOutputOp: String = _tfWhisperSignatures.getOrElse(
      ModelSignatureConstants.LogitsOutput.key,
      ModelSignatureConstants.LogitsOutput.value)
  }

  private object OnnxSignatures {
    val encoderInputKey: String = "input_features"
    val encoderOutputKey = "last_hidden_state"
    val encoderStateOutputKeys: Array[String] = Array(
      "present.0.encoder.key",
      "present.0.encoder.value",
      "present.1.encoder.key",
      "present.1.encoder.value",
      "present.2.encoder.key",
      "present.2.encoder.value",
      "present.3.encoder.key",
      "present.3.encoder.value")

    val decoderOutputKey: String = "logits"
    val decoderStateOutputKeys: Array[String] = Array(
      "present.0.decoder.key",
      "present.0.decoder.value",
      "present.1.decoder.key",
      "present.1.decoder.value",
      "present.2.decoder.key",
      "present.2.decoder.value",
      "present.3.decoder.key",
      "present.3.decoder.value")
  }

  /** @param audios
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
      audios: Seq[AnnotationAudio],
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

    val logitProcessors: LogitProcessorList = {
      val processorList = new LogitProcessorList()

      // Assuming bos token at the front, get the last forced token
      val generationBeginIdx: Int = forcedDecoderIds match {
        case Some(forcedTokens) => 1 + forcedTokens.maxBy(_._1)._1
        case None => 1
      }

      if (beginSuppressTokens.isDefined)
        processorList.addProcess(
          new SuppressLogitProcessor(beginSuppressTokens.get, Some(generationBeginIdx)))

      if (suppressTokenIds.isDefined)
        processorList.addProcess(new SuppressLogitProcessor(suppressTokenIds.get))

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

      // Task token should be at index index 2. If None, then it should be "transcribe" (by default forced?)
      totalForcedDecoderIds =
        if (task.isDefined) totalForcedDecoderIds.updated(2, vocabWithAddedTokens(task.get))
        else totalForcedDecoderIds

      processorList.addProcess(new ForcedTokenLogitProcessor(totalForcedDecoderIds.toArray))

      processorList
    }

    val batchedAudio = audios.grouped(batchSize).toArray

    val batchDecodedIds =
      batchedAudio.flatMap { batch: Seq[AnnotationAudio] =>
        val featuresBatch = batch.map { case AnnotationAudio(_, rawFloats, _) =>
          preprocessor.extractFeatures(rawFloats)
        }.toArray

        val batchDecoderStartIds = Array.fill(batchedAudio.length, 1)(bosTokenId)

        val tokenIds: Array[Array[Int]] = detectedEngine match {
          case TensorFlow.name =>
            val session =
              tensorflowWrapper.get
                .getTFSessionWithSignature(configProtoBytes, savedSignatures = signatures)

            val encodedBatchFeatures: Tensor =
              encode(featuresBatch, Some(session), None).asInstanceOf[Tensor]

            // Generate the tokens
            val tokenIds: Array[Array[Int]] = generate(
              encodedBatchFeatures,
              batchDecoderStartIds,
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
              randomSeed,
              session,
              logitProcessors)

            tfTensorResources.clearTensors()
            encodedBatchFeatures.close()

            tokenIds
          case ONNX.name =>
            if (beamSize > 1)
              logger.warn(
                "Currently the Whisper ONNX model only supports greedy search. Will default to this behavior.")

            val (encoderSession, env) = onnxWrappers.get.encoder.getSession()
            val decoderSession = onnxWrappers.get.decoder.getSession()._1
            val decoderWithPastSession = onnxWrappers.get.decoderWithPast.getSession()._1

            val encodedBatchTensor: OnnxTensor =
              encode(featuresBatch, None, Some((encoderSession, env))).asInstanceOf[OnnxTensor]

            val (logits, initEncoderStates, initDecoderStates) =
              initOnnxDecoder(
                batchDecoderStartIds,
                encodedBatchTensor,
                logitProcessors,
                (decoderSession, env))

            encodedBatchTensor.close()

            val batchInitGenerationTokenIds: Array[Array[Int]] =
              logits.map { logitsArray =>
                Array(bosTokenId, argmax(logitsArray))
              }

            val tokenIds = generateGreedyOnnx(
              batchInitGenerationTokenIds,
              replaceStateKeys(initEncoderStates),
              replaceStateKeys(initDecoderStates),
              maxOutputLength,
              minOutputLength,
              logitProcessors,
              (decoderWithPastSession, env))

            tokenIds
        }

        decode(tokenIds)
      }

    var sentBegin, nextSentEnd = 0
    batchDecodedIds.zip(audios).map { case (content, audio) =>
      nextSentEnd += content.length - 1
      val annotation = new Annotation(
        annotatorType = AnnotatorType.DOCUMENT,
        begin = sentBegin,
        end = nextSentEnd,
        result = content,
        metadata = audio.metadata)
      sentBegin += nextSentEnd + 1
      annotation
    }
  }

  /** Decode a sequence of sentences
    *
    * @param sentences
    *   Sequence of sentences
    * @return
    *   Sequence of decoded sentences
    */
  def decode(sentences: Array[Array[Int]]): Seq[String] = {
    sentences.map(s => tokenDecoder.decodeTokens(s))
  }

  /** Encodes a batch of preprocessed input audio.
    *
    * @param features
    *   Batch of Whisper features
    * @return
    *   Tensor with encoded features for each batch
    */

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
          .feed(TfSignatures.encoderInputOp, featuresTensors)
          .fetch(TfSignatures.encoderOutputOp)
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

  /** Get model output for a batch of input sequences
    *
    * TODO: Caching
    *
    * @param encodedInputsTensor
    *   Batch of encoded features as a Tensor
    * @param decoderInputIds
    *   Batch of decoder input ids
    * @param maxLength
    *   Max length of the output
    * @param session
    *   tensorflow session
    * @return
    *   Model output logits for the last input token for the batches
    */
  def getModelOutput(
      encodedInputsTensor: Tensor,
      decoderInputIds: Seq[Array[Int]],
      maxLength: Int,
      session: Session): Array[Array[Float]] = {

    val truncatedInputIds = decoderInputIds.map(_.slice(0, maxLength))

    val decoderInputIdsTensor: Tensor =
      tfTensorResources.createTensor[Array[Array[Int]]](truncatedInputIds.toArray)

    val runner = session.runner
      .feed(TfSignatures.decoderInputIdsOp, decoderInputIdsTensor)
      .feed(TfSignatures.decoderEncoderOutputsOp, encodedInputsTensor)
      .fetch(TfSignatures.decoderOutputOp)

    val decoderOuts = runner.run().asScala
    val logitsRaw = tensorflow.TensorResources.extractFloats(decoderOuts.head)
    decoderOuts.foreach(_.close())

    val nextTokenLogits =
      logitsRaw.grouped(logitsSize).toArray // Should result in length batch size

    tfTensorResources.clearTensors()
    nextTokenLogits
  }

  def generate(
      decoderEncoderStateTensors: Tensor,
      decoderInputIds: Array[Array[Int]],
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
      session: Session,
      logitProcessorList: LogitProcessorList): Array[Array[Int]] = {
    val dummyEncoderInput =
      Seq.fill(decoderInputIds.length)(Array.empty[Int]) // Needs to be size of batch
    val dummyEncoderAttentionMaskTensors: Tensor = null // not needed

    if (beamSize == 1) // Equivalent to greedy search
      super.generateGreedy(
        encoderInputIds = dummyEncoderInput,
        decoderEncoderStateTensors = decoderEncoderStateTensors,
        encoderAttentionMaskTensors = dummyEncoderAttentionMaskTensors,
        decoderInputs = decoderInputIds,
        maxOutputLength = maxOutputLength,
        minOutputLength = minOutputLength,
        vocabSize = logitsSize,
        eosTokenId = eosTokenId,
        paddingTokenId = paddingTokenId,
        applySoftmax = false,
        session = session,
        logitProcessor = Some(logitProcessorList))
    else
      super.generate(
        inputIds = dummyEncoderInput,
        decoderEncoderStateTensors = decoderEncoderStateTensors,
        encoderAttentionMaskTensors = dummyEncoderAttentionMaskTensors,
        decoderInputs = decoderInputIds,
        maxOutputLength = maxOutputLength,
        minOutputLength = minOutputLength,
        doSample = doSample,
        beamSize = beamSize,
        numReturnSequences = numReturnSequences,
        temperature = temperature,
        topK = topK,
        topP = topP,
        repetitionPenalty = repetitionPenalty,
        noRepeatNgramSize = noRepeatNgramSize,
        vocabSize = logitsSize,
        eosTokenId = eosTokenId,
        paddingTokenId = paddingTokenId,
        randomSeed = randomSeed,
        session = session,
        applySoftmax = false)
  }

  def getModelOutput(
      encoderInputIds: Seq[Array[Int]],
      decoderInputIds: Seq[Array[Int]],
      decoderEncoderStateTensors: Tensor,
      encoderAttentionMaskTensors: Tensor,
      maxLength: Int,
      session: Session): Array[Array[Float]] = {
    getModelOutput(decoderEncoderStateTensors, decoderInputIds, maxLength, session)
  }

  private def initOnnxDecoder(
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
      sessionOutput.getFloatArray(OnnxSignatures.decoderOutputKey).grouped(logitsSize).toArray

    val logits = logitProcessors.process(decoderInputIds, rawLogits, decoderInputIds.head.length)

    val encoderStates =
      sessionOutput.getOnnxTensors(OnnxSignatures.encoderStateOutputKeys)

    val decoderStates =
      sessionOutput.getOnnxTensors(OnnxSignatures.decoderStateOutputKeys)

    (logits, encoderStates, decoderStates)
  }

  private def getOnnxDecoderOutput(
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
      sessionOutput.getOnnxTensors(OnnxSignatures.decoderStateOutputKeys)

    lastTokensTensor.close()

    val batchLogits = logits.grouped(logitsSize).toArray
    (batchLogits, updatedDecoderStates)
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
    * @param minOutputLength
    *   min output length
    * @return
    */
  private def generateGreedyOnnx(
      initInputIds: Array[Array[Int]],
      encoderStateTensors: Map[String, OnnxTensor],
      decoderStateTensors: Map[String, OnnxTensor],
      maxOutputLength: Int,
      minOutputLength: Int,
      logitProcessor: LogitProcessorList,
      onnxSession: (OrtSession, OrtEnvironment)): Array[Array[Int]] = {

    var generatedIds: Array[Array[Int]] = initInputIds
    var currentDecoderStateTensors = decoderStateTensors

    while (!greedyGenerationFinished(generatedIds, eosTokenId, maxOutputLength)) {

      val (rawBatchLogits: Array[Array[Float]], updatedDecoderStates: Map[String, OnnxTensor]) =
        getOnnxDecoderOutput(
          generatedIds,
          encoderStateTensors,
          currentDecoderStateTensors,
          onnxSession)

      currentDecoderStateTensors.foreach { case (_, tensor) =>
        tensor.close()
      }

      val batchLogits =
        logitProcessor.process(generatedIds, rawBatchLogits, generatedIds.head.length)
      val nextTokenIds: Array[Int] = batchLogits.map(argmax)
      currentDecoderStateTensors = replaceStateKeys(updatedDecoderStates)

      generatedIds =
        generatedIds.zip(nextTokenIds).map { case (currentIds: Array[Int], nextId: Int) =>
          currentIds ++ Array(nextId)
        }
    }

    generatedIds
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

  def replaceStateKeys(outputs: Map[String, OnnxTensor]): Map[String, OnnxTensor] =
    outputs.map { case (key, t) =>
      (key.replace("present", "past_key_values"), t)
    }
}
