package com.johnsnowlabs.ml.ai.seq2seq

import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import com.johnsnowlabs.nlp.annotators.tokenizer.normalizer.MosesPunctNormalizer
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

import scala.collection.JavaConverters._

private[johnsnowlabs] abstract class MarianEncoderDecoder(
    val sppSrc: SentencePieceWrapper,
    val sppTrg: SentencePieceWrapper)
    extends Serializable {

  private val langCodeRe = ">>.+<<".r

  def sessionWarmup(): Unit = {
    val dummyInput = Array.fill(1)(0)
    tag(Seq(dummyInput), 0, 0, 0, 1)
  }

  def decode(sentences: Array[Array[Int]], vocabsArray: Array[String]): Seq[String] = {

    sentences.map { s =>
      val filteredPads = s.filter(x => x != 0)
      val pieceTokens = filteredPads.map { pieceId =>
        vocabsArray(pieceId)
      }
      sppTrg.getSppModel.decodePieces(pieceTokens.toList.asJava)
    }

  }

  def encode(
      sentences: Seq[Annotation],
      normalizer: MosesPunctNormalizer,
      maxSeqLength: Int,
      vocabsArray: Array[String],
      langId: Int,
      unknownTokenId: Int,
      eosTokenId: Int): Seq[Array[Int]] = {

    sentences.map { s =>
      // remove language code from the source text
      val sentWithoutLangId = langCodeRe.replaceFirstIn(s.result, "").trim
      val normalizedSent = normalizer.normalize(sentWithoutLangId)
      val pieceTokens =
        sppSrc.getSppModel.encodeAsPieces(normalizedSent).toArray.map(x => x.toString)

      val pieceIds = pieceTokens.map { piece =>
        val pieceId = vocabsArray.indexOf(piece)
        if (pieceId > 0) {
          pieceId
        } else {
          unknownTokenId
        }
      }

      if (langId > 0)
        Array(langId) ++ pieceIds.take(maxSeqLength - 2) ++ Array(eosTokenId)
      else
        pieceIds.take(maxSeqLength - 1) ++ Array(eosTokenId)
    }

  }

  protected def tag(
      batch: Seq[Array[Int]],
      maxOutputLength: Int,
      paddingTokenId: Int,
      eosTokenId: Int,
      vocabSize: Int,
      doSample: Boolean = false,
      temperature: Double = 1.0d,
      topK: Int = 50,
      topP: Double = 1.0d,
      repetitionPenalty: Double = 1.0d,
      noRepeatNgramSize: Int = 0,
      randomSeed: Option[Long] = None,
      ignoreTokenIds: Array[Int] = Array()): Array[Array[Int]]

  /** generate seq2seq via encoding, generating, and decoding
    *
    * @param sentences
    *   none empty Annotation
    * @param batchSize
    *   size of baches to be process at the same time
    * @param maxInputLength
    *   maximum length for input
    * @param maxOutputLength
    *   maximum length for output
    * @param vocabs
    *   list of all vocabs
    * @param langId
    *   language id for multi-lingual models
    * @return
    */
  def predict(
      sentences: Seq[Annotation],
      batchSize: Int = 1,
      maxInputLength: Int,
      maxOutputLength: Int,
      vocabs: Array[String],
      langId: String,
      doSample: Boolean = false,
      temperature: Double = 1.0d,
      topK: Int = 50,
      topP: Double = 1.0d,
      repetitionPenalty: Double = 1.0d,
      noRepeatNgramSize: Int = 0,
      randomSeed: Option[Long] = None,
      ignoreTokenIds: Array[Int] = Array()): Array[Annotation] = {

    val normalizer = new MosesPunctNormalizer()

    val paddingTokenId = vocabs.indexOf("<pad>")
    val unknownTokenId = vocabs.indexOf("<unk>")
    val eosTokenId = vocabs.indexOf("</s>")
    val ignoreTokenIdsWithPadToken = ignoreTokenIds ++ Array(paddingTokenId)
    val vocabSize = vocabs.toSeq.length

    val langIdPieceId = if (langId.nonEmpty) {
      vocabs.indexOf(langId)
    } else {
      val lang = langCodeRe.findFirstIn(sentences.head.result.trim).getOrElse(-1L)
      vocabs.indexOf(lang)
    }

    val batchDecoder = sentences.grouped(batchSize).toArray.flatMap { batch =>
      val batchSP = encode(
        batch,
        normalizer,
        maxInputLength,
        vocabs,
        langIdPieceId,
        unknownTokenId,
        eosTokenId)
      val spIds = tag(
        batch = batchSP,
        maxOutputLength = maxOutputLength,
        paddingTokenId = paddingTokenId,
        eosTokenId = eosTokenId,
        vocabSize = vocabSize,
        doSample = doSample,
        temperature = temperature,
        topK = topK,
        topP = topP,
        repetitionPenalty = repetitionPenalty,
        noRepeatNgramSize = noRepeatNgramSize,
        randomSeed = randomSeed,
        ignoreTokenIds = ignoreTokenIdsWithPadToken)
      decode(spIds, vocabs)
    }

    var sentBegin, nextSentEnd = 0
    batchDecoder.zip(sentences).map { case (content, sent) =>
      nextSentEnd += content.length - 1
      val annotations = new Annotation(
        annotatorType = AnnotatorType.DOCUMENT,
        begin = sentBegin,
        end = nextSentEnd,
        result = content,
        metadata = sent.metadata)
      sentBegin += nextSentEnd + 1
      annotations
    }
  }

}
