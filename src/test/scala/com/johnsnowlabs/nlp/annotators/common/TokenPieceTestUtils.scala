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

package com.johnsnowlabs.nlp.annotators.common

private[johnsnowlabs] object TokenPieceTestUtils {

  /** A word-start piece whose `.token` and `.wordpiece` are the same text -- the common case of a
    * word that fits in a single vocabulary entry.
    */
  def piece(wordpiece: String, isWordStart: Boolean): TokenPiece =
    TokenPiece(
      wordpiece = wordpiece,
      token = wordpiece,
      pieceId = 0,
      isWordStart = isWordStart,
      begin = 0,
      end = 0)

  /** A continuation piece of a multi-piece word. Every real tokenizer (WordpieceEncoder.encode,
    * BpeTokenizer.getTokenPieces) sets `.token` to the WHOLE ORIGINAL WORD on every piece of that
    * word, not to the piece's own text -- so a word-start piece and its continuation piece(s)
    * always share the same `.token` value. Use this (not `piece`) whenever a test models more
    * than one piece of the same word under the vocab strategy, or it isn't representative of what
    * joinWordPieces actually receives.
    */
  def continuationPiece(wordpiece: String, wholeWordToken: String): TokenPiece =
    TokenPiece(
      wordpiece = wordpiece,
      token = wholeWordToken,
      pieceId = 0,
      isWordStart = false,
      begin = 0,
      end = 0)
}
