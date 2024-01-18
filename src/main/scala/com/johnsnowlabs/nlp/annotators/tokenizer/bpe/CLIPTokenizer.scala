/*
 * Copyright 2017-2023 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, TokenPiece}

import scala.util.matching.Regex

class CLIPTokenizer(
    merges: Map[(String, String), Int],
    vocab: Map[String, Int],
    specialTokens: SpecialTokens)
    extends Gpt2Tokenizer(
      merges,
      vocab,
      specialTokens,
      padWithSequenceTokens = true,
      prependString = "",
      addPrefixSpaceToSentence = false,
      alwaysAddPrefix = false) {

  private val wordEnding = "</w>"

  // Case insensitive and does not include white spaces, adapted from transformers
  override val splitPattern: Regex =
    raw"""(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+""".r

  /** CLIP Specific tokenization. We append "<\w>" to word ends.
    *
    * @return
    *   Array of TokenPieces, corresponding to encoded token
    */
  override protected def bpe(indToken: IndexedToken): Array[TokenPiece] = {
    var processedToken = ""
    try {
      processedToken = preProcessTokenForBpe(indToken.token)

      var word: Array[String] = Array[String]()
      // split the word into characters, to be combined into subwords
      word = processedToken.map(_.toString).toArray
      val pairs: Array[(String, String)] = getBytePairs(word)

      if (pairs.isEmpty)
        word = Array(processedToken + wordEnding)
      else {
        word.update(word.length - 1, word.last + wordEnding)
        pairs.update(pairs.length - 1, (pairs.last._1, pairs.last._2 + wordEnding))
        word = performMerges(word, pairs)
      }

      getTokenPieces(indToken, word)
    } catch {
      case _: java.util.NoSuchElementException =>
        Array(
          TokenPiece(
            indToken.token,
            indToken.token,
            specialTokens.unk.id,
            isWordStart = true,
            indToken.begin,
            indToken.end))
    }
  }

  override def tokenizeSubText(text: String, indexOffset: Int): Array[IndexedToken] = {
    splitPattern
      .findAllMatchIn(text)
      .map(tok => IndexedToken(tok.matched, tok.start + indexOffset, tok.end + indexOffset - 1))
      .toArray
  }

}
