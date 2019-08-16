package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import scala.collection.Map


case class WordpieceEmbeddingsSentence
(
  tokens: Array[TokenPieceEmbeddings],
  sentenceId: Int,
  sentenceEmbeddings: Option[Array[Float]] = None
)

case class TokenPieceEmbeddings(wordpiece: String, token: String, pieceId: Int,
                                isWordStart: Boolean, isOOV: Boolean,
                                embeddings: Array[Float], begin: Int, end: Int)

object TokenPieceEmbeddings {
  def apply(piece: TokenPiece, embeddings: Array[Float]): TokenPieceEmbeddings = {
    TokenPieceEmbeddings(
      wordpiece = piece.wordpiece,
      token = piece.token,
      pieceId = piece.pieceId,
      isWordStart = piece.isWordStart,
      isOOV = false, // FIXME: I think BERT wont have OOV, this "constructor" is called from TensorFlowBert
      embeddings = embeddings,
      begin = piece.begin,
      end = piece.end)
  }
  def apply(wordpiece: String, token: String, pieceId: Int,
            isWordStart: Boolean,
            embeddings: Option[Array[Float]], zeroArray: Array[Float], begin: Int, end: Int): TokenPieceEmbeddings = {

    val vector = embeddings.getOrElse(zeroArray)
    val oov = embeddings match { case Some(_) => false; case default => true; }
    TokenPieceEmbeddings(
      wordpiece = wordpiece,
      token = token,
      pieceId = pieceId,
      isWordStart = isWordStart,
      isOOV = oov,
      embeddings = vector,
      begin = begin,
      end = end)
  }
}

object  WordpieceEmbeddingsSentence extends Annotated[WordpieceEmbeddingsSentence] {
  override def annotatorType: String = AnnotatorType.WORD_EMBEDDINGS

  override def unpack(annotations: Seq[Annotation]): Seq[WordpieceEmbeddingsSentence] = {
    val tokens = annotations
      .filter(_.annotatorType == annotatorType)
      .groupBy(_.metadata("sentence").toInt)

    tokens.map{case (idx: Int, sentenceTokens: Seq[Annotation]) =>
      val sentenceEmbeddings = sentenceTokens.map(t => t.sentence_embeddings).headOption
      val tokensWithSentence = sentenceTokens.map { token =>
        new TokenPieceEmbeddings(
          wordpiece = token.result,
          token = token.metadata("token"),
          pieceId = token.metadata("pieceId").toInt,
          isWordStart = token.metadata("isWordStart").toBoolean,
          isOOV = token.metadata("isOOV").toBoolean,
          embeddings = token.embeddings,
          begin = token.begin,
          end = token.end
        )
      }.toArray

      WordpieceEmbeddingsSentence(tokensWithSentence, idx, sentenceEmbeddings)
    }.toSeq.sortBy(_.sentenceId)
  }

  override def pack(sentences: Seq[WordpieceEmbeddingsSentence]): Seq[Annotation] = {
    sentences.zipWithIndex.flatMap{case (sentence, sentenceIndex) =>
      var isFirstToken = true
      sentence.tokens.map{token =>
        // Store embeddings for token
        val embeddings = token.embeddings

        // Store sentence embeddings only in one token
        val sentenceEmbeddings =
          if (isFirstToken && sentence.sentenceEmbeddings.isDefined)
            sentence.sentenceEmbeddings.get
          else
            Array.emptyFloatArray

        isFirstToken = false
        Annotation(annotatorType, token.begin, token.end, token.token,
          Map("sentence" -> sentenceIndex.toString,
            "token" -> token.token,
            "pieceId" -> token.pieceId.toString,
            "isWordStart" -> token.isWordStart.toString,
            "isOOV" -> token.isOOV.toString
          ),
          embeddings,
          sentenceEmbeddings
        )
      }
    }
  }
}
