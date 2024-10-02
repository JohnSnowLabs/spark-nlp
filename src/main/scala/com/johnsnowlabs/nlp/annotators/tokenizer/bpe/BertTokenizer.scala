/*
 * Copyright 2017-2024 John Snow Labs
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

import com.johnsnowlabs.nlp.annotators.common.WordpieceTokenizedSentence
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.BasicTokenizer

import java.nio.charset.Charset
import scala.collection.mutable.ListBuffer

class BertTokenizer(val vocab: Map[String, Int], val specialTokens: SpecialTokens)
    extends BasicTokenizer {

  /** Encode the input sequence to indexes IDs adding padding where necessary */
  def encode(
      sentences: Seq[(WordpieceTokenizedSentence, Int)],
      maxSequenceLength: Int): Seq[Array[Int]] = {
    val maxSentenceLength =
      Array(
        maxSequenceLength - 2,
        sentences.map { case (wpTokSentence, _) =>
          wpTokSentence.tokens.length
        }.max).min

    sentences
      .map { case (wpTokSentence, _) =>
        val tokenPieceIds = wpTokSentence.tokens.map(t => t.pieceId)
        val padding = Array.fill(maxSentenceLength - tokenPieceIds.length)(specialTokens.pad.id)

        Array(specialTokens.sentenceStart.id) ++ tokenPieceIds.take(maxSentenceLength) ++ Array(
          specialTokens.sentenceEnd.id) ++ padding
      }
  }

  def decodeTokens(tokens: Array[Int]): String = {
    val specialTokens = SpecialTokens.getSpecialTokensForModel("bert", vocab)
    val decoderVocab: Map[Int, String] = vocab.map(x => (x._2, x._1))
    val unicodeToByteMapping: Map[String, Int] =
      bytesToUnicodeMapping.map(x => (x._2, x._1))
    val text = tokens
      .map(token => decoderVocab.getOrElse(token, ""))
      .filter(x => !specialTokens.contains(x))
      .mkString("")
    val bytes = text.map(x => unicodeToByteMapping(x.toString)).map(x => x.toByte).toArray
    new String(bytes, Charset.forName("UTF-8"))
  }

  /** Mapping for bytes to a different set of unicode characters (especially white spaces). This
    * improved model performance for gpt-2
    */
  protected val bytesToUnicodeMapping: Map[Int, String] = {
    val bytes: ListBuffer[Int] =
      ListBuffer.range('!', '~' + 1) ++ ListBuffer.range('¡', '¬' + 1) ++ ListBuffer
        .range('®', 'ÿ' + 1)
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

}
