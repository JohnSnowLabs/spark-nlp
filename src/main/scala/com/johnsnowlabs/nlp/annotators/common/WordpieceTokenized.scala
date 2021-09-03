/*
 * Copyright 2017-2021 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
