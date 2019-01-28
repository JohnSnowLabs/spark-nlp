package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import scala.collection.Map


object WordpieceTokenized extends Annotated[WordpieceTokenizedSentence] {

  override def annotatorType: String = AnnotatorType.WORDPIECE

  override def unpack(annotations: Seq[Annotation]): Seq[WordpieceTokenizedSentence] = {
    val tokens = annotations
      .filter(_.annotatorType == annotatorType)
      .toArray

    SentenceSplit.unpack(annotations).map(sentence => {
      tokens.filter(token =>
        token.begin >= sentence.start & token.end <= sentence.end
      ).map(token =>
        TokenPiece(wordpiece = token.result,
           token = token.metadata("token"),
           pieceId = token.metadata("pieceId").toInt,
           isWordStart = token.metadata("isWordStart").toBoolean,
           begin = token.begin,
           end = token.end
        )
      )
    }).filter(_.nonEmpty).map(tokens => WordpieceTokenizedSentence(tokens))

  }

  override def pack(sentences: Seq[WordpieceTokenizedSentence]): Seq[Annotation] = {
    var sentenceIndex = 0

    sentences.flatMap{sentence =>
      sentenceIndex += 1
      sentence.tokens.map{token =>
        Annotation(annotatorType, token.begin, token.end, token.wordpiece,
          Map("sentence" -> sentenceIndex.toString,
            "isWordStart" -> token.isWordStart.toString,
            "pieceId" -> token.pieceId.toString,
            "token" -> token.token)
        )
      }}
  }
}

case class WordpieceTokenizedSentence(tokens: Array[TokenPiece])
case class TokenPiece(wordpiece: String, token: String, pieceId: Int, isWordStart: Boolean, begin: Int, end: Int)
