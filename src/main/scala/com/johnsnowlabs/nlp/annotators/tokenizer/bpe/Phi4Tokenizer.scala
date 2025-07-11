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

package com.johnsnowlabs.nlp.annotators.tokenizer.bpe

import com.johnsnowlabs.nlp.annotators.common.IndexedToken

import java.nio.charset.Charset
import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex

/** Tokenizer for Microsoft Phi-4 model (https://huggingface.co/microsoft/phi-4)
  *
  * This class implements the byte pair encoding (BPE) tokenizer logic for the phi-4 model.
  */
class Phi4Tokenizer(
    merges: Map[(String, String), Int],
    vocab: Map[String, Int],
    specialTokens: SpecialTokens,
    padWithSequenceTokens: Boolean = true,
    prependString: String = "",
    addPrefixSpaceToSentence: Boolean = false,
    alwaysAddPrefix: Boolean = true,
    splitPatternRegex: Regex =
      raw"""(?i)(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""".r)
    extends BpeTokenizer(
      merges,
      vocab,
      specialTokens,
      padWithSequenceTokens,
      addPrefixSpaceToSentence,
      alwaysAddPrefix) {

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

  override val prefixForPieceId: Option[String] =
    if (prependString.nonEmpty) Some(prependString) else None

  protected val decoderVocab: Map[Int, String] = vocab.map(x => (x._2, x._1))

  protected val unicodeToByteMapping: Map[String, Int] =
    bytesToUnicodeMapping.map(x => (x._2, x._1))

  override def preProcessTokenForBpe(token: String): String = {
    token
      .getBytes("UTF-8")
      .map { b => if (b < 0) 256 + b else b }
      .foldLeft("")(_ + bytesToUnicodeMapping(_))
  }

  val splitPattern: Regex = splitPatternRegex

  override def tokenizeSubText(text: String, indexOffset: Int): Array[IndexedToken] = {
    splitPattern
      .findAllMatchIn(
        if (prefixForPieceId.isDefined || text.startsWith(" ")) text
        else " " + text)
      .map(tok => IndexedToken(tok.matched, tok.start + indexOffset, tok.end + indexOffset - 1))
      .toArray
  }

  def decodeTokens(tokens: Array[Int]): String = {
    val text = tokens
      .map(token => decoderVocab(token))
      .filter(x => !specialTokens.contains(x))
      .mkString("")

    val bytes =
      text.map(x => unicodeToByteMapping(x.toString)).map(x => x.toByte).toArray
    new String(bytes, Charset.forName("UTF-8"))
  }
}
