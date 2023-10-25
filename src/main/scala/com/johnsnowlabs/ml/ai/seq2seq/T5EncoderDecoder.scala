package com.johnsnowlabs.ml.ai.seq2seq

import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

import scala.collection.mutable
import scala.math.exp

abstract class T5EncoderDecoder(
    val spp: SentencePieceWrapper,
    val additionalTokens: Map[Int, String] = Map()) {

  protected val paddingTokenId = 0
  protected val eosTokenId = 1
  protected val pieceSize: Int = spp.getSppModel.getPieceSize
  protected val vocabSize = 32128

  def sessionWarmup(): Unit = {
    val dummyInput = Array.fill(1)(0) ++ Array(eosTokenId)

    tag(
      Seq(dummyInput),
      maxNewTokens = 1,
      maxTextLength = 1,
      doSample = false,
      temperature = 0f,
      topK = 0,
      topP = 0f,
      repetitionPenalty = 0f,
      noRepeatNgramSize = 0,
      randomSeed = Option(0L),
      stopAtEos = true,
      ignoreTokenIds = Array(0))
  }

  protected def decode(sentences: Array[Array[Int]]): Seq[String] = {

    sentences.map { s =>
      val filteredPieceIds = s.filter(x => x <= pieceSize || additionalTokens.contains(x))
      val additionalTokenPositions =
        filteredPieceIds.zipWithIndex.filter(x => additionalTokens.contains(x._1)).map(_._2)
      val decodedStrings = if (additionalTokenPositions.nonEmpty) {
        var offset = 0
        additionalTokenPositions.map(i => {
          val slice =
            spp.getSppModel.decodeIds(filteredPieceIds.slice(offset, i): _*) + additionalTokens(
              filteredPieceIds(i))
          offset = i + 1
          slice
        }) ++ Array(
          spp.getSppModel.decodeIds(filteredPieceIds.slice(offset, filteredPieceIds.length): _*))
      } else {
        Array(spp.getSppModel.decodeIds(filteredPieceIds: _*))
      }
      decodedStrings.mkString("")
    }

  }

  protected def encode(sentences: Seq[Annotation], task: String): Seq[Array[Int]] = {
    sentences.map(s => {
      val sentWithTask = if (task.nonEmpty) task.concat(" ").concat(s.result) else s.result
      spp.getSppModel.encodeAsIds(sentWithTask) ++ Array(this.eosTokenId)
    })
  }

  protected def tag(
      batch: Seq[Array[Int]],
      maxNewTokens: Int,
      maxTextLength: Int,
      doSample: Boolean,
      topK: Int,
      topP: Double,
      temperature: Double,
      noRepeatNgramSize: Int,
      repetitionPenalty: Double,
      randomSeed: Option[Long],
      ignoreTokenIds: Array[Int] = Array(),
      stopAtEos: Boolean): Array[Array[Int]]

  def predict(
      sentences: Seq[Annotation],
      task: String,
      batchSize: Int,
      maxNewTokens: Int,
      maxTextLength: Int,
      doSample: Boolean,
      topK: Int,
      topP: Double,
      temperature: Double,
      randomSeed: Option[Long] = None,
      ignoreTokenIds: Array[Int] = Array(),
      isCaseSensitive: Boolean,
      stopAtEos: Boolean,
      noRepeatNgramSize: Int,
      repetitionPenalty: Double): Seq[Annotation] = {

    val batchDecoder = sentences.grouped(batchSize).toArray.flatMap { batch =>
      val batchSP = encode(batch, task)
      val spIds = tag(
        batch = batchSP,
        maxNewTokens = maxNewTokens,
        maxTextLength = maxTextLength,
        doSample = doSample,
        topK = topK,
        topP = topP,
        temperature = temperature,
        randomSeed = randomSeed,
        ignoreTokenIds = ignoreTokenIds,
        stopAtEos = stopAtEos,
        noRepeatNgramSize = noRepeatNgramSize,
        repetitionPenalty = repetitionPenalty)
      decode(spIds)

    }

    var sentBegin, nextSentEnd = 0
    batchDecoder.zip(sentences).map { case (content, sent) =>
      nextSentEnd += content.length - 1
      val newAnnotation = new Annotation(
        annotatorType = AnnotatorType.DOCUMENT,
        begin = sentBegin,
        end = nextSentEnd,
        result = content,
        metadata = sent.metadata)
      sentBegin += nextSentEnd + 1
      newAnnotation
    }
  }

  def generate(
      prompts: Seq[Annotation],
      batchSize: Int,
      maxNewTokens: Int,
      maxContextLength: Int,
      doSample: Boolean,
      topK: Int,
      topP: Double,
      temperature: Double,
      randomSeed: Option[Long],
      ignoreTokenIds: Array[Int],
      isCaseSensitive: Boolean,
      stopAtEos: Boolean,
      noRepeatNgramSize: Int,
      repetitionPenalty: Double): Seq[Annotation] = {
    predict(
      sentences = prompts,
      task = "",
      batchSize = batchSize,
      maxNewTokens = maxNewTokens,
      maxTextLength = maxContextLength,
      doSample = doSample,
      topK = topK,
      topP = topP,
      temperature = temperature,
      randomSeed = randomSeed,
      ignoreTokenIds = ignoreTokenIds,
      isCaseSensitive = isCaseSensitive,
      stopAtEos = stopAtEos,
      noRepeatNgramSize = noRepeatNgramSize,
      repetitionPenalty = repetitionPenalty)
  }
}
