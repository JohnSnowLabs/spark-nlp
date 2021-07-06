/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence, TokenPiece}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex

class RobertaTokenizer(
                        merges: Map[(String, String), Int],
                        vocab: Map[String, Int],
                        specialTokens: SpecialTokens,
                        padWithSentenceTokens: Boolean = false
                      ) extends BpeTokenizer(merges, vocab, specialTokens, padWithSentenceTokens) {

  /**
    * Mapping for bytes to a different set of unicode characters (especially white spaces).
    * This improved model performance for gpt-2
    */
  private val bytesToUnicodeMapping: Map[Int, String] = {
    val bytes: ListBuffer[Int] = ListBuffer.range(
      '!',
      '~' + 1
    ) ++ ListBuffer.range('¡', '¬' + 1) ++ ListBuffer.range('®', 'ÿ' + 1)
    val characters: ListBuffer[Int] = bytes.clone
    var n = 0
    for (b <- 0 to 256) {
      if (!bytes.contains(b)) {
        bytes += b
        characters += (256 + n)
        n += 1
      }
    }
    (bytes zip characters.map(_.toChar.toString)).toMap
  }
  override val prependForPieceId: Option[String] = Some("Ġ")

  override def preProcessTokenForBpe(token: String): String =
    token.foldLeft("")(_ + bytesToUnicodeMapping(_))

  val splitPattern: Regex = raw"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""".r

  override def tokenizeSubText(text: String, indexOffset: Int): Array[IndexedToken] = {
    // split pattern based on gpt2's bpe tokenizer
    splitPattern
      .findAllMatchIn(text)
      .map(tok => IndexedToken(tok.matched, tok.start + indexOffset, tok.end + indexOffset - 1))
      .toArray
  }
}
