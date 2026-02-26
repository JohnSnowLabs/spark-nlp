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

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, TokenPiece}

import java.nio.charset.Charset
import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex

class ModernBertTokenizer(
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

  /** Converts a raw string token to its byte-level unicode representation. Optionally prepends a
    * space (which maps to 'Ġ') before encoding,
    */
  override def preProcessTokenForBpe(token: String): String = {
    token
      .getBytes("UTF-8")
      .map { b => if (b < 0) 256 + b else b }
      .foldLeft("")(_ + bytesToUnicodeMapping(_))
  }

  /** Encode a token using byte-level BPE. If the token starts with a space (prepended by the
    * caller to signal word-boundary), that space is byte-encoded to 'Ġ' and included in the BPE
    * run The leading space is then stripped from the stored wordpiece for offset alignment.
    */
  override protected def bpe(indToken: IndexedToken): Array[TokenPiece] = {
    val hasPrefixSpace = indToken.token.startsWith(" ")
    try {
      val processedToken = preProcessTokenForBpe(indToken.token)

      var word: Array[String] = processedToken.map(_.toString).toArray
      val pairs = getBytePairs(word)
      if (pairs.isEmpty) word = Array(processedToken)
      else word = performMerges(word, pairs)

      // For a token with a synthetic leading space the first BPE piece starts with 'Ġ';
      // we strip it for the wordpiece text and adjust the offset accordingly.
      var currentIndex = indToken.begin
      var firstPiece = true
      word.map { subWord =>
        val isWordStart = firstPiece
        firstPiece = false

        val origSubWord =
          if (isWordStart && hasPrefixSpace && subWord.startsWith("\u0120"))
            subWord.substring(1) // strip the synthetic Ġ from stored wordpiece
          else subWord

        // Decode back through unicodeToByteMapping to get the length in original chars
        val origBytes = origSubWord.map(c => unicodeToByteMapping(c.toString).toByte).toArray
        val origLen = new String(origBytes, Charset.forName("UTF-8")).length
        val startIndex = currentIndex
        currentIndex = startIndex + origLen

        // Vocab lookup uses the full BPE piece (including Ġ)
        val subWordId = vocab.getOrElse(subWord, specialTokens.unk.id)
        TokenPiece(
          origSubWord,
          indToken.token.trim(),
          subWordId,
          isWordStart,
          startIndex,
          currentIndex - 1)
      }
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
