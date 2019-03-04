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
                                isWordStart: Boolean,
                                embeddings: Array[Float], begin: Int, end: Int)

object TokenPieceEmbeddings {
  def apply(piece: TokenPiece, embeddings: Array[Float]): TokenPieceEmbeddings = {
    TokenPieceEmbeddings(
      wordpiece = piece.wordpiece,
      token = piece.token,
      pieceId = piece.pieceId,
      isWordStart = piece.isWordStart,
      embeddings = embeddings,
      begin = piece.begin,
      end = piece.end)
  }
}

object WordpieceEmbeddingsSentence extends Annotated[WordpieceEmbeddingsSentence] {
  override def annotatorType: String = AnnotatorType.WORD_EMBEDDINGS

  override def unpack(annotations: Seq[Annotation]): Seq[WordpieceEmbeddingsSentence] = {
    val tokens = annotations
      .filter(_.annotatorType == annotatorType)
      .toArray

    SentenceSplit.unpack(annotations).zipWithIndex.map{case (sentence: Sentence, idx: Int) =>
      val sentenceTokens = tokens.filter(token =>
        token.begin >= sentence.start & token.end <= sentence.end
      )

      val sentenceEmbeddings = sentenceTokens.flatMap(t => t.calculations.get("sentence_embeddings")).headOption

      val tokensWithSentence = sentenceTokens.map { token =>
        new TokenPieceEmbeddings(
          wordpiece = token.result,
          token = token.metadata("token"),
          pieceId = token.metadata("pieceId").toInt,
          isWordStart = token.metadata("isWordStart").toBoolean,
          embeddings = token.getCalculations("embeddings"),
          begin = token.begin,
          end = token.end
        )
      }

      WordpieceEmbeddingsSentence(tokensWithSentence, idx, sentenceEmbeddings)
    }
  }

  override def pack(sentences: Seq[WordpieceEmbeddingsSentence]): Seq[Annotation] = {
    sentences.zipWithIndex.flatMap{case (sentence, sentenceIndex) =>
      var isFirstToken = true
      sentence.tokens.map{token =>
        // Store embeddings for token
        val embeddings = Map("embeddings" -> token.embeddings)

        // Store sentence embeddings only in one token
        val calculations =
          if (isFirstToken && sentence.sentenceEmbeddings.isDefined)
            embeddings ++ Map("sentence_embeddings" -> sentence.sentenceEmbeddings.get)
          else
            embeddings

        isFirstToken = false
        Annotation(annotatorType, token.begin, token.end, token.wordpiece,
          Map("sentence" -> sentenceIndex.toString,
            "token" -> token.token,
            "pieceId" -> token.pieceId.toString,
            "isWordStart" -> token.isWordStart.toString
          ),
          calculations
        )
      }
    }
  }
}
