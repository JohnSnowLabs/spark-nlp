/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.ml.tensorflow.sentencepiece

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence, TokenPiece}

/** @param spp
  *   StencePieceWrapper loaded from either from disk or a saved Spark NLP model
  * @param caseSensitive
  *   whether it cares about uppercase or lowercases
  * @param delimiterId
  *   what is the part prefix id
  * @param pieceIdFromZero
  *   whether or not pieceId should be as is or plus 1
  */
private[ml] class SentencepieceEncoder(
    spp: SentencePieceWrapper,
    caseSensitive: Boolean,
    delimiterId: Int = 13,
    pieceIdOffset: Int = 0) {

  /** @param token
    *   IndexedToken input that is used for encoding to piece tokens and piece ids
    * @return
    *   Array of TokenPiece which are piece tokens with piece ids
    */
  def encode(token: IndexedToken): Array[TokenPiece] = {

    val text = token.token
    var start = 0
    var end = text.length
    val normalizedDelimiterId = delimiterId + pieceIdOffset

    val tokenContent = if (caseSensitive) token.token else token.token.toLowerCase()
    val wordPieces = spp.getSppModel.encodeAsPieces(tokenContent).toArray.map(x => x.toString)
    val encodedIds = spp.getSppModel.encodeAsIds(tokenContent)
    val pieceIds = encodedIds.map(x => x + pieceIdOffset)
    wordPieces.zip(pieceIds).filter(id => id._2 != normalizedDelimiterId).map { piece =>
      val tokenPiece =
        TokenPiece(piece._1, token.token, piece._2, start == 0, token.begin + start, token.end)
      start = end
      end = text.length
      tokenPiece
    }
  }

  def encodeSentence(sentence: Sentence, maxLength: Int): Array[TokenPiece] = {

    val text = sentence.content.take(maxLength)
    var start = 0
    var end = 0
    val normalizedDelimiterId = delimiterId + pieceIdOffset

    val sentContent = if (caseSensitive) text else text.toLowerCase()
    val wordPieces = spp.getSppModel.encodeAsPieces(sentContent).toArray.map(x => x.toString)
    val encodedIds = spp.getSppModel.encodeAsIds(sentContent)
    val delimiterPiece = spp.getSppModel.idToPiece(normalizedDelimiterId)
    val pieceIds = encodedIds.map(x => x + pieceIdOffset)
    wordPieces.zip(pieceIds).filter(id => id._2 != normalizedDelimiterId).map { piece =>
      val cleanToken = piece._1.replace(delimiterPiece, "")
      end = end + cleanToken.length - 1
      val tokenPiece =
        TokenPiece(
          piece._1,
          cleanToken,
          piece._2,
          piece._1.startsWith(delimiterPiece),
          start,
          end)
      start = end + 1
      tokenPiece
    }
  }
}
