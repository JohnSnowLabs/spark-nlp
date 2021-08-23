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


case class WordpieceEmbeddingsSentence
(
  tokens: Array[TokenPieceEmbeddings],
  sentenceId: Int
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
      val tokensWithSentence = sentenceTokens.map { token =>
        new TokenPieceEmbeddings(
          wordpiece = token.result,
          token = token.metadata("token"),
          pieceId = token.metadata("pieceId").toInt,
          isWordStart = token.metadata("isWordStart").toBoolean,
          isOOV = token.metadata.getOrElse("isOOV", "false").toBoolean,
          embeddings = token.embeddings,
          begin = token.begin,
          end = token.end
        )
      }.toArray

      WordpieceEmbeddingsSentence(tokensWithSentence, idx)
    }.toSeq.sortBy(_.sentenceId)
  }

  override def pack(sentences: Seq[WordpieceEmbeddingsSentence]): Seq[Annotation] = {
    sentences.flatMap{sentence =>
      var isFirstToken = true
      sentence.tokens.map{ token =>
        // Store embeddings for token
        val embeddings = token.embeddings

        isFirstToken = false
        Annotation(annotatorType, token.begin, token.end, token.token,
          Map("sentence" -> sentence.sentenceId.toString,
            "token" -> token.token,
            "pieceId" -> token.pieceId.toString,
            "isWordStart" -> token.isWordStart.toString,
            "isOOV" -> token.isOOV.toString
          ),
          embeddings
        )
      }
    }
  }
}
