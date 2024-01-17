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
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.onnx.TensorResources.implicits._
import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import com.johnsnowlabs.nlp.Annotation
import scala.collection.JavaConverters._
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT

private[johnsnowlabs] class LLAMA2(
    val onnxWrappers: DecoderWrappers,
    val spp: SentencePieceWrapper,
    generationConfig: GenerationConfig)
    extends Serializable {

  private val GenerationConfig(
    bosTokenId: Int,
    _,
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

  /** Encode a sequence of sentences
    * @param sentences
    *   Sequence of sentences
    * @param task
    *   Task
    * @return
    *   Sequence of encoded sentences
    */
  def encode(sentences: Seq[Annotation]): Seq[Array[Int]] = {
    sentences.map(s => {
      val sentWithTask = s.result
      spp.getSppModel.encodeAsIds(sentWithTask)
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
      maxInputLength: Int): Array[Array[Int]] = {
    val (encoderSession, env) = onnxWrappers.decoder.getSession()
    val ignoreTokenIdsInt = ignoreTokenIds
    val expandedEncoderInputIdsVals =
      batch.flatMap(x => List.fill(beamSize)(x.take(maxInputLength)))
    val sequencesLength = expandedEncoderInputIdsVals.map(x => x.length).toArray
    val maxSentenceLength = sequencesLength.max // - curLen

    val numReturn_sequences = 1
    // from config

    var effectiveBatch_size = 1
    var effectiveBatch_mult = 1
    if (doSample) {
      effectiveBatch_size = expandedEncoderInputIdsVals.length * numReturn_sequences
      effectiveBatch_mult = numReturn_sequences
    } else {
      effectiveBatch_size = expandedEncoderInputIdsVals.length
      effectiveBatch_mult = 1
    }

    // Run the prompt through the decoder and get the past
    val decoderOutputs =
      generateGreedyOnnx(
        expandedEncoderInputIdsVals.toArray,
        (encoderSession, env),
        maxOutputLength)
    decoderOutputs
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
      maxInputLength: Int): Seq[Annotation] = {

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
        maxInputLength)

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

  private def getDecoderOutputs(
      inputIds: Array[Array[Int]],
      onnxSession: (OrtSession, OrtEnvironment)): (Array[Array[Float]]) = {
    val (session, env) = onnxSession

    val inputIdsLong: Array[Array[Long]] =
      inputIds.map { tokenIds => tokenIds.map(_.toLong) }

    val inputIdsLongTensor: OnnxTensor =
      OnnxTensor.createTensor(env, inputIdsLong)
    val decoderAttentionMask: OnnxTensor =
      OnnxTensor.createTensor(env, inputIdsLong.map(_.map(_ => 1L)))
    val decoderInputs: java.util.Map[String, OnnxTensor] = Map(
      OnnxSignatures.decoderInputIDs -> inputIdsLongTensor,
      OnnxSignatures.decoderAttentionMask -> decoderAttentionMask).asJava
    val sessionOutput = session.run(decoderInputs)
    val logits = sessionOutput.getFloatArray(OnnxSignatures.decoderOutput)
    inputIdsLongTensor.close()
    val batchLogits = logits.grouped(vocabSize).toArray
    batchLogits

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
      nextTokenIds.foreach(x => println(s"new ids:$x"))
      generatedIds =
        generatedIds.zip(nextTokenIds).map { case (currentIds: Array[Int], nextId: Int) =>
          currentIds ++ Array(nextId)
        }
      // print lens of generatedIds
      generatedIds.foreach(x => println(x.length))
    }
    generatedIds
  }

  private object OnnxSignatures {
    val decoderInputIDs: String = "input_ids"
    val decoderAttentionMask: String = "attention_mask"
    // create decoder past for 32 layers of key and value eg. past_key_values.0.key and past_key_values.0.value
    val decoderPast: Array[String] = (0 until 32)
      .flatMap(i => Seq(s"past_key_values.$i.key", s"past_key_values.$i.value"))
      .toArray
    val decoderOutput: String = "logits"
    val decoderPresent: Array[String] =
      (0 until 32).flatMap(i => Seq(s"present.$i.key", s"present.$i.value")).toArray
  }

}
