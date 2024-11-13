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
import com.johnsnowlabs.ml.ai.util.Generation.{Generate, GenerationConfig}
import com.johnsnowlabs.ml.onnx.OnnxSession
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.onnx.TensorResources.implicits._
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.annotators.common.SentenceSplit
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{
  BpeTokenizer,
  LLAMA3Tokenizer,
  SpecialTokens
}
import org.intel.openvino.InferRequest
import org.tensorflow.{Session, Tensor}

import scala.collection.JavaConverters._

private[johnsnowlabs] class CoHere(
    val onnxWrappers: Option[DecoderWrappers],
    val openvinoWrapper: Option[OpenvinoWrapper],
    merges: Map[(String, String), Int],
    vocabulary: Map[String, Int],
    addedTokens: Map[String, Int],
    generationConfig: GenerationConfig)
    extends Serializable
    with Generate {

  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions
  val detectedEngine: String =
    if (onnxWrappers.isDefined) ONNX.name
    else if (openvinoWrapper.isDefined) Openvino.name
    else ONNX.name
  private var nextPositionId: Option[Array[Long]] = None
  private val GenerationConfig(
    bosTokenId: Int,
    paddingTokenId: Int,
    eosTokenId: Int,
    vocabSize: Int,
    beginSuppressTokens,
    suppressTokenIds,
    forcedDecoderIds) =
    generationConfig

  val reversedVocabulary: Map[Int, String] = vocabulary.map(_.swap)
  val specialTokens: SpecialTokens = SpecialTokens(
    vocabulary,
    startTokenString = reversedVocabulary(bosTokenId),
    endTokenString = reversedVocabulary(eosTokenId),
    unkTokenString = reversedVocabulary(eosTokenId),
    maskTokenString = reversedVocabulary(eosTokenId),
    padTokenString = reversedVocabulary(eosTokenId),
    additionalStrings = addedTokens.keys.toArray)

  val bpeTokenizer: LLAMA3Tokenizer = BpeTokenizer
    .forModel(
      "llama3",
      merges = merges,
      vocab = vocabulary,
      specialTokens = Some(specialTokens),
      addPrefixSpaceToSentence = true)
    .asInstanceOf[LLAMA3Tokenizer]

  /** Decode a sequence of sentences
    * @param sentences
    *   Sequence of sentences
    * @return
    *   Sequence of decoded sentences
    */
  def decode(sentences: Array[Array[Int]]): Seq[String] = {
    sentences.map(s => bpeTokenizer.decodeTokens(s.map(_.toInt)))
  }

  /** Encode a sequence of sentences
    * @param sentences
    *   Sequence of sentences
    * @return
    *   Sequence of encoded sentences
    */
  def encode(sentences: Seq[Annotation]): Seq[Array[Int]] = {
    SentenceSplit
      .unpack(sentences)
      .map(s => {
        val sentWithTask = s
        bpeTokenizer
          .tokenize(sentWithTask)
          .map(bpeTokenizer.encode)
          .flatMap(_.map(_.pieceId))
      })
  }

  def tag(
      batch: Seq[Array[Int]],
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
      stopTokenIds: Array[Int]): Array[Array[Int]] = {
    val ignoreTokenIdsInt = ignoreTokenIds
    val expandedDecoderInputsVals = batch
    val sequencesLength = expandedDecoderInputsVals.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max // - curLen

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

    // Run the prompt through the decoder and get the past
//    val decoderOutputs =
//      generateGreedyOnnx(
//        expandedDecoderInputsVals.toArray,
//        (encoderSession, env),
//        maxOutputLength)
    val (decoderEncoderStateTensors, encoderAttentionMaskTensors, session) =
      detectedEngine match {
        case ONNX.name =>
          // dummy tensors for decoder encode state and attention mask
          val (encoderSession, env) = onnxWrappers.get.decoder.getSession(onnxSessionOptions)
          (
            Right(OnnxTensor.createTensor(env, Array(0))),
            Right(OnnxTensor.createTensor(env, Array(1))),
            Right((env, encoderSession)))
        case Openvino.name =>
          // not needed
          (null, null, null)
      }
    val ovInferRequest: Option[InferRequest] = detectedEngine match {
      case ONNX.name => None
      case Openvino.name => Some(openvinoWrapper.get.getCompiledModel().create_infer_request())
    }
    // output with beam search
    val modelOutputs = generate(
      batch,
      decoderEncoderStateTensors,
      encoderAttentionMaskTensors,
      expandedDecoderInputsVals.toArray,
      maxOutputLength + maxSentenceLength,
      minOutputLength,
      doSample,
      beamSize,
      1,
      temperature,
      topK,
      topP,
      repetitionPenalty,
      noRepeatNgramSize,
      this.vocabSize,
      this.eosTokenId,
      this.paddingTokenId,
      randomSeed,
      ignoreTokenIdsInt,
      session,
      applySoftmax = false,
      ovInferRequest = ovInferRequest,
      stopTokenIds = stopTokenIds)

//    decoderOutputs
    modelOutputs
  }

  def predict(
      sentences: Seq[Annotation],
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
      maxInputLength: Int,
      stopTokenIds: Array[Int]): Seq[Annotation] = {

    val batchDecoder = sentences.grouped(batchSize).toArray.flatMap { batch =>
      val batchSP = encode(batch)
      val spIds = tag(
        batchSP,
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
        stopTokenIds)

      decode(spIds)

    }

    var sentBegin, nextSentEnd = 0
    val annotations = batchDecoder.zip(sentences).map { case (content, sent) =>
      nextSentEnd += content.length - 1
      val annots = new Annotation(
        annotatorType = DOCUMENT,
        begin = sentBegin,
        end = nextSentEnd,
        result = content,
        metadata = sent.metadata)
      sentBegin += nextSentEnd + 1
      annots
    }
    annotations
  }

  private def getDecoderOutputsWithPast(
      inputIds: Array[Array[Int]],
      decoderPast: Map[String, OnnxTensor],
      onnxSession: (OrtSession, OrtEnvironment))
      : (Array[Array[Float]], Map[String, OnnxTensor]) = {
    val (session, env) = onnxSession

    val lastTokens: Array[Array[Long]] =
      inputIds.map { tokenIds =>
        Array(tokenIds.last.toLong)
      }

    val lastTokensTensor: OnnxTensor =
      OnnxTensor.createTensor(env, lastTokens)
    val decoderAttentionMask: OnnxTensor =
      OnnxTensor.createTensor(env, lastTokens.map(_.map(_ => 1L)))
    val decoderWithPastInputs: java.util.Map[String, OnnxTensor] = (Map(
      OnnxSignatures.decoderInputIDs -> lastTokensTensor,
      OnnxSignatures.decoderAttentionMask -> decoderAttentionMask) ++ decoderPast).asJava
    val sessionOutput = session.run(decoderWithPastInputs)
    val logits = sessionOutput.getFloatArray(OnnxSignatures.decoderOutput)
    val decoderPresent = sessionOutput.getOnnxTensors(OnnxSignatures.decoderPresent)
    lastTokensTensor.close()
    val batchLogits = logits.grouped(vocabSize).toArray
    (batchLogits, decoderPresent)

  }

  override def getModelOutput(
      encoderInputIds: Seq[Array[Int]],
      decoderInputIds: Seq[Array[Int]],
      decoderEncoderStateTensors: Either[Tensor, OnnxTensor],
      encoderAttentionMaskTensors: Either[Tensor, OnnxTensor],
      maxLength: Int,
      session: Either[Session, (OrtEnvironment, OrtSession)],
      ovInferRequest: Option[InferRequest]): Array[Array[Float]] = {
    detectedEngine match {
      case TensorFlow.name =>
        // not implemented yet
        Array()
      case ONNX.name =>
        val (env, decoderSession) = session.right.get
        val decoderOutputs =
          getDecoderOutputs(decoderInputIds.toArray, onnxSession = (decoderSession, env))
        decoderOutputs
      case Openvino.name =>
        val decoderOutputs =
          getDecoderOutputsOv(
            encoderInputIds.toArray,
            decoderInputIds.toArray,
            ovInferRequest.get)
        decoderOutputs
    }
  }

  private def getDecoderOutputsOv(
      encoderInputIds: Array[Array[Int]],
      decoderInputIds: Array[Array[Int]],
      inferRequest: InferRequest): (Array[Array[Float]]) = {

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

    val inputIdsLongTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(shape, inputIdsLong)
    val decoderAttentionMask: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize, decoderInputIds.head.length), attentionMask)
    val decoderPositionIDs: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(shape, inputPositionIDsLong)
    val beamIdxTensor: org.intel.openvino.Tensor =
      new org.intel.openvino.Tensor(Array(batchSize), beamIdx)

    inferRequest.set_tensor(OpenVinoSignatures.decoderInputIDs, inputIdsLongTensor)
    inferRequest.set_tensor(OpenVinoSignatures.decoderAttentionMask, decoderAttentionMask)
    inferRequest.set_tensor(OpenVinoSignatures.decoderPositionIDs, decoderPositionIDs)
    inferRequest.set_tensor(OpenVinoSignatures.decoderBeamIdx, beamIdxTensor)

    inferRequest.infer()

    val result = inferRequest.get_tensor(OpenVinoSignatures.decoderOutput)
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
  private def getDecoderOutputs(
      inputIds: Array[Array[Int]],
      onnxSession: (OrtSession, OrtEnvironment)): (Array[Array[Float]]) = {
    val (session, env) = onnxSession

    val inputIdsLong: Array[Array[Long]] =
      inputIds.map { tokenIds => tokenIds.map(_.toLong) }

    val inputPositionIDsLong: Array[Array[Long]] =
      inputIds.map { tokenIds =>
        tokenIds.zipWithIndex.map { case (_, i) =>
          i.toLong
        }
      }

    val inputIdsLongTensor: OnnxTensor =
      OnnxTensor.createTensor(env, inputIdsLong)
    val decoderAttentionMask: OnnxTensor =
      OnnxTensor.createTensor(env, inputIdsLong.map(_.map(_ => 1L)))
    val decoderPositionIDs: OnnxTensor =
      OnnxTensor.createTensor(env, inputPositionIDsLong)

    val decoderInputs: java.util.Map[String, OnnxTensor] = Map(
      OnnxSignatures.decoderInputIDs -> inputIdsLongTensor,
      OnnxSignatures.decoderAttentionMask -> decoderAttentionMask,
      OnnxSignatures.decoderPositionIDs -> decoderPositionIDs).asJava
    val sessionOutput = session.run(decoderInputs)

    val sequenceLength = inputIds.head.length
    val batchSize = inputIds.length

//    val logits = sessionOutput.getFloatArray(OnnxSignatures.decoderOutput)
//    inputIdsLongTensor.close()
//    decoderPositionIDs.close()
//    decoderAttentionMask.close()
//    val batchLogits = logits.grouped(vocabSize).toArray
//    batchLogits

    val logitsRaw = sessionOutput.getFloatArray(OnnxSignatures.decoderOutput)
    val decoderOutputs = (0 until batchSize).map(i => {
      logitsRaw
        .slice(
          i * sequenceLength * vocabSize + (sequenceLength - 1) * vocabSize,
          i * sequenceLength * vocabSize + sequenceLength * vocabSize)
    })
    decoderOutputs.toArray
  }

  /** Gets the index with the highest score
    *
    * @param scores
    *   Array of Scores to max
    * @return
    *   Index of the highest score
    */
  private def argmax(scores: Array[Float]): Int =
    scores.zipWithIndex.maxBy { case (score, _) =>
      score
    }._2
  private def greedyGenerationFinished(
      decoderIds: Seq[Array[Int]],
      eosTokenId: Int,
      maxOutputLength: Int): Boolean =
    decoderIds.map(_.last).forall(_ == eosTokenId) || decoderIds.head.length == maxOutputLength

  private def generateGreedyOnnx(
      inputIds: Array[Array[Int]],
      onnxSession: (OrtSession, OrtEnvironment),
      maxOutputLength: Int): (Array[Array[Int]]) = {

    val sequencesLength = inputIds.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max // - curLen
    var generatedIds: Array[Array[Int]] = inputIds
    while (!greedyGenerationFinished(
        generatedIds,
        eosTokenId,
        maxOutputLength + maxSentenceLength)) {

      val (batchLogits: Array[Array[Float]]) =
        Array(getDecoderOutputs(generatedIds, onnxSession).last)

      val nextTokenIds: Array[Int] = batchLogits.map(argmax)
      generatedIds =
        generatedIds.zip(nextTokenIds).map { case (currentIds: Array[Int], nextId: Int) =>
          currentIds ++ Array(nextId)
        }
    }
    generatedIds
  }

  private object OnnxSignatures {
    val decoderInputIDs: String = "input_ids"
    val decoderAttentionMask: String = "attention_mask"
    val decoderPositionIDs: String = "position_ids"

    // create decoder past for 32 layers of key and value eg. past_key_values.0.key and past_key_values.0.value
    val decoderPast: Array[String] = (0 until 32)
      .flatMap(i => Seq(s"past_key_values.$i.key", s"past_key_values.$i.value"))
      .toArray
    val decoderOutput: String = "logits"
    val decoderPresent: Array[String] =
      (0 until 32).flatMap(i => Seq(s"present.$i.key", s"present.$i.value")).toArray
  }

  private object OpenVinoSignatures {
    val encoderInputIDs: String = "input_ids"
    val encoderAttentionMask: String = "attention_mask"

    val encoderOutput: String = "last_hidden_state"

    val decoderInputIDs: String = "input_ids"
    val decoderEncoderAttentionMask: String = "encoder_attention_mask"
    val decoderAttentionMask: String = "attention_mask"
    val decoderPositionIDs: String = "position_ids"
    val decoderBeamIdx: String = "beam_idx"
    val decoderEncoderState: String = "encoder_hidden_states"

    val decoderOutput: String = "logits"
  }
}
