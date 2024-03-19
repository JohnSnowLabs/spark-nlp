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
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWithoutPastWrappers
import com.johnsnowlabs.ml.onnx.TensorResources.implicits._
import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import org.tensorflow.{Session, Tensor}

import scala.collection.JavaConverters._

private[johnsnowlabs] class NLLB(
    val onnxWrappers: EncoderDecoderWithoutPastWrappers,
    val spp: SentencePieceWrapper,
    generationConfig: GenerationConfig,
    vocab: Map[String, Int])
    extends Serializable
    with Generate {

  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

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
  private val reverseVocab = vocab.map(_.swap)

  /** Decode a sequence of sentences
    * @param sentences
    *   Sequence of sentences
    * @return
    *   Sequence of decoded sentences
    */
  def decode(sentences: Array[Array[Int]]): Seq[String] = {
    sentences.map { s =>
      val filteredPieceIds = s.filter(x => x < (vocabSize - 205))
      val filteredPieces = filteredPieceIds.map(x => reverseVocab.getOrElse(x, ""))
      val sentence = spp.getSppModel.decodePieces(filteredPieces.toList.asJava)
      sentence
    }
  }

  /** Encode a sequence of sentences
    * @param sentences
    *   Sequence of sentences
    * @return
    *   Sequence of encoded sentences
    */
  def encode(sentences: Seq[Annotation]): Seq[Array[Int]] = {
    val encodedPieces = sentences.map(s => {
      val sentWithTask = s.result
      spp.getSppModel.encodeAsPieces(sentWithTask).toArray.map(x => x.toString)
    })
    val encodedIds = encodedPieces.map(p => {
      p.map(x => vocab.getOrElse(x, 0))
    })
    encodedIds
  }

  /** Translates a batch of sentences from a source language to a target language
    * @param batch
    *   a batch of sentences to translate
    * @param minOutputLength
    *   minimum length of the output
    * @param maxOutputLength
    *   maximum length of the output
    * @param doSample
    *   whether to sample or not
    * @param temperature
    *   temperature for sampling
    * @param topK
    *   topK for sampling
    * @param topP
    *   topP for sampling
    * @param repetitionPenalty
    *   repetition penalty for sampling
    * @param noRepeatNgramSize
    *   no repeat ngram size for sampling
    * @param randomSeed
    *   random seed for sampling
    * @param ignoreTokenIds
    *   token ids to ignore
    * @param beamSize
    *   beam size for beam search
    * @param maxInputLength
    *   maximum length of the input
    * @param srcLangToken
    *   source language token
    * @param tgtLangToken
    *   target language token
    * @return
    */
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
      srcLangToken: Int,
      tgtLangToken: Int): Array[Array[Int]] = {
    val (encoderSession, encoderEnv) = onnxWrappers.encoder.getSession(onnxSessionOptions)
    val (decoderSession, decoderEnv) = onnxWrappers.decoder.getSession(onnxSessionOptions)
    val ignoreTokenIdsInt = ignoreTokenIds
    val expandedEncoderInputsVals =
      batch.flatMap(x => List.fill(beamSize)(x.take(maxInputLength))).toArray
    val sequencesLength = expandedEncoderInputsVals.map(x => x.length)
    val maxSentenceLength = sequencesLength.max // - curLen

    expandedEncoderInputsVals.zipWithIndex.foreach { case (input, i) =>
      expandedEncoderInputsVals(i) =
        Array(vocabSize + srcLangToken - 205) ++ input ++ Array(eosTokenId)
    }

    val decoderInputIds: Array[Array[Int]] =
      batch.map(_ => Array(eosTokenId, vocabSize + tgtLangToken - 205)).toArray

    val numReturn_sequences = 1
    // from config

    var effectiveBatch_size = 1
    var effectiveBatch_mult = 1

    if (doSample) {
      effectiveBatch_size = expandedEncoderInputsVals.length * numReturn_sequences
      effectiveBatch_mult = numReturn_sequences
    } else {
      effectiveBatch_size = expandedEncoderInputsVals.length
      effectiveBatch_mult = 1
    }

    // run encoder
    val decoderEncoderStateTensors =
      getEncoderOutput(expandedEncoderInputsVals, Right((encoderEnv, encoderSession)))

    val encoderAttentionMaskTensors =
      Right(
        OnnxTensor
          .createTensor(decoderEnv, expandedEncoderInputsVals.toArray.map(_.map(_ => 1L))))

    // output with beam search
    val modelOutputs = generate(
      batch,
      decoderEncoderStateTensors,
      encoderAttentionMaskTensors,
      decoderInputIds,
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
      Right((decoderEnv, decoderSession)),
      applySoftmax = false)

    // Run the prompt through the decoder and get the past
    //    val decoderOutputs =
    //      generateGreedyOnnx(
    //        decoderInputIds,
    //        decoderEncoderStateTensors,
    //        encoderAttentionMaskTensors,
    //        onnxSession = (decoderSession, decoderEnv))

    // close sessions
    decoderEncoderStateTensors.fold(
      tfTensor => {
        // not implemented yet
      },
      onnxTensor => onnxTensor.close())

    encoderAttentionMaskTensors.fold(
      tfTensor => {
        // not implemented yet
      },
      onnxTensor => onnxTensor.close())

    encoderEnv.close()
    decoderEnv.close()

    //    decoderOutputs
    modelOutputs
  }

  /** Translates a batch of sentences from a source language to a target language
    * @param sentences
    *   a batch of sentences to translate
    * @param batchSize
    *   batch size
    * @param minOutputLength
    *   minimum length of the output
    * @param maxOutputLength
    *   maximum length of the output
    * @param doSample
    *   whether to sample or not
    * @param temperature
    *   temperature for sampling
    * @param topK
    *   topK for sampling
    * @param topP
    *   topP for sampling
    * @param repetitionPenalty
    *   repetition penalty for sampling
    * @param noRepeatNgramSize
    *   no repeat ngram size for sampling
    * @param randomSeed
    *   random seed for sampling
    * @param ignoreTokenIds
    *   token ids to ignore
    * @param beamSize
    *   beam size for beam search
    * @param maxInputLength
    *   maximum length of the input
    * @param srcLangToken
    *   source language token
    * @param tgtLangToken
    * @return
    */
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
      srcLangToken: Int,
      tgtLangToken: Int): Seq[Annotation] = {

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
        srcLangToken,
        tgtLangToken)
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

  /** Generates a sequence of tokens using beam search
    * @param encoderInputIds
    *   Input IDs for the Encoder
    * @param session
    *   Tensorflow/ONNX Session
    * @return
    *   Last hidden state of the encoder
    */
  private def getEncoderOutput(
      encoderInputIds: Seq[Array[Int]],
      session: Either[Session, (OrtEnvironment, OrtSession)]): Either[Tensor, OnnxTensor] = {
    session.fold(
      tfSession => {
        // not implemented yet
        null
      },
      onnxSession => {

        val (env, encoderSession) = onnxSession

        val encoderAttentionMask: OnnxTensor =
          OnnxTensor.createTensor(env, encoderInputIds.toArray.map(_.map(_ => 1L)))

        val encoderInputTensors: OnnxTensor =
          OnnxTensor.createTensor(env, encoderInputIds.toArray.map(_.map(_.toLong)))

        val encoderInputs: java.util.Map[String, OnnxTensor] = Map(
          OnnxSignatures.encoderInputIDs -> encoderInputTensors,
          OnnxSignatures.encoderAttentionMask -> encoderAttentionMask).asJava

        val encoderResults = encoderSession.run(encoderInputs)

        val encoderStateBuffer =
          try {
            val encoderStateTensor = encoderResults
              .get(OnnxSignatures.encoderOutput)
              .get()
              .asInstanceOf[OnnxTensor]

            val shape = encoderStateTensor.getInfo.getShape
            encoderStateTensor.getFloatBuffer
              .array()
              .grouped(shape(2).toInt)
              .toArray
              .grouped(shape(1).toInt)
              .toArray
          } finally {
            if (encoderResults != null) encoderResults.close()
          }

        encoderInputTensors.close()
        encoderAttentionMask.close()

        val encoderStateTensors = OnnxTensor.createTensor(env, encoderStateBuffer)

        Right(encoderStateTensors)
      })
  }

  /** Gets the model output
    * @param encoderInputIds
    *   Input IDs for the Encoder
    * @param decoderInputIds
    *   Input IDs for the Decoder
    * @param decoderEncoderStateTensors
    *   Tensor of encoded input for the decoder
    * @param encoderAttentionMaskTensors
    *   Tensor for encoder attention mask
    * @param maxLength
    *   Max length of the input
    * @param session
    *   Tensorflow/ONNX Session
    * @return
    *   Logits for the input
    */
  override def getModelOutput(
      encoderInputIds: Seq[Array[Int]],
      decoderInputIds: Seq[Array[Int]],
      decoderEncoderStateTensors: Either[Tensor, OnnxTensor],
      encoderAttentionMaskTensors: Either[Tensor, OnnxTensor],
      maxLength: Int,
      session: Either[Session, (OrtEnvironment, OrtSession)]): Array[Array[Float]] = {

    session.fold(
      tfSession => {
        // not implemented yet
        Array()
      },
      onnxSession => {
        val (env, decoderSession) = onnxSession
        val decoderOutputs =
          getDecoderOutputs(
            decoderInputIds.toArray,
            decoderEncoderStateTensors,
            encoderAttentionMaskTensors,
            onnxSession = (decoderSession, env))
        decoderOutputs
      })

  }

  /** Gets the decoder outputs
    * @param inputIds
    *   input ids
    * @param decoderEncoderStateTensors
    *   decoder encoder state tensors
    * @param encoderAttentionMaskTensors
    *   encoder attention mask tensors
    * @param onnxSession
    *   onnx session
    * @return
    *   decoder outputs
    */
  private def getDecoderOutputs(
      inputIds: Array[Array[Int]],
      decoderEncoderStateTensors: Either[Tensor, OnnxTensor],
      encoderAttentionMaskTensors: Either[Tensor, OnnxTensor],
      onnxSession: (OrtSession, OrtEnvironment)): (Array[Array[Float]]) = {
    val (session, env) = onnxSession

    val inputIdsLong: Array[Array[Long]] =
      inputIds.map { tokenIds => tokenIds.map(_.toLong) }

    val inputIdsLongTensor: OnnxTensor =
      OnnxTensor.createTensor(env, inputIdsLong)

    val encoderAttentionMaskTensor = encoderAttentionMaskTensors.fold(
      tfTensor => {
        // not implemented yet
        null
      },
      onnxTensor => onnxTensor)

    val decoderEncoderStateTensor = decoderEncoderStateTensors.fold(
      tfTensor => {
        // not implemented yet
        null
      },
      onnxTensor => onnxTensor)

    val decoderInputs: java.util.Map[String, OnnxTensor] = Map(
      OnnxSignatures.decoderInputIDs -> inputIdsLongTensor,
      OnnxSignatures.decoderEncoderAttentionMask -> encoderAttentionMaskTensor,
      OnnxSignatures.decoderEncoderState -> decoderEncoderStateTensor).asJava
    val sessionOutput = session.run(decoderInputs)

    val sequenceLength = inputIds.head.length
    val batchSize = inputIds.length

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
      decoderInputIds: Array[Array[Int]],
      decoderEncoderStateTensors: Either[Tensor, OnnxTensor],
      encoderAttentionMaskTensors: Either[Tensor, OnnxTensor],
      session: Either[Session, (OrtEnvironment, OrtSession)]): (Array[Array[Int]]) = {

    val sequencesLength = decoderInputIds.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max // - curLen
    var generatedIds: Array[Array[Int]] = Array()

    while (!greedyGenerationFinished(generatedIds, eosTokenId, maxSentenceLength)) {

      session.fold(
        tfSession => {
          // not implemented yet
          Array()
        },
        onnxSession => {
          val (env, decoderSession) = onnxSession
          val decoderOutputs =
            getDecoderOutputs(
              decoderInputIds.toArray,
              decoderEncoderStateTensors,
              encoderAttentionMaskTensors,
              onnxSession = (decoderSession, env))

          val nextTokenIds: Array[Int] = decoderOutputs.map(argmax)
          generatedIds =
            generatedIds.zip(nextTokenIds).map { case (currentIds: Array[Int], nextId: Int) =>
              currentIds ++ Array(nextId)
            }
        })
    }
    generatedIds
  }

  private object OnnxSignatures {
    val encoderInputIDs: String = "input_ids"
    val encoderAttentionMask: String = "attention_mask"

    val encoderOutput: String = "last_hidden_state"

    val decoderInputIDs: String = "input_ids"
    val decoderEncoderAttentionMask: String = "encoder_attention_mask"
    val decoderEncoderState: String = "encoder_hidden_states"

    val decoderOutput: String = "logits"
  }

}
