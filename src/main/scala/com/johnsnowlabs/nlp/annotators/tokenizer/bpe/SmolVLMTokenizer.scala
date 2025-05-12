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

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence}

import java.nio.charset.Charset
import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex
import scala.collection.mutable

class SmolVLMTokenizer(
    merges: Map[(String, String), Int],
    vocab: Map[String, Int],
    specialTokens: SpecialTokens,
    padWithSequenceTokens: Boolean = true,
    prependString: String = "",
    addPrefixSpaceToSentence: Boolean = false,
    alwaysAddPrefix: Boolean = false,
    splitPatternRegex: Regex =
      raw"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""".r)
    extends BpeTokenizer(
      merges,
      vocab,
      specialTokens,
      padWithSequenceTokens,
      addPrefixSpaceToSentence,
      alwaysAddPrefix) {

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

  // Differs from Transformers, space is always prepended.
  // FIX: Space should not be prepended to all tokens, but to the beginning of the text only. Otherwise token
  // such as '.' get space prepended and they should not.
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
    // split pattern based on gpt2's bpe tokenizer
    splitPattern
      .findAllMatchIn(text) // Remove conditional space prepending
      .map(tok => IndexedToken(tok.matched, tok.start + indexOffset, tok.end + indexOffset - 1))
      .toArray
  }

  //  def decodeTokens(tokens: Array[Int]): String = {
  //    val decoded = new mutable.StringBuilder()
  //    tokens.foreach { token =>
  //      {
  //        val decodedToken = decoderVocab(token)
  //        if (!specialTokens.contains(decodedToken)) {
  //          if (decodedToken.startsWith("<0x") && decodedToken.endsWith(">")) {
  //            val strippedHex = decodedToken.replaceAll("<0x|>", "")
  //            val byteValue = Integer.parseInt(strippedHex, 16)
  //            decoded.append(byteValue.toChar)
  //          } else {
  //            decoded.append(decodedToken)
  //          }
  //        }
  //      }
  //
  //    }
  //    decoded.toString().replaceAll(decoderVocab(29871), " ").trim()
  //  }

  override def tokenize(sentence: Sentence): Array[IndexedToken] = {
    var originalText = sentence.content
    if (originalText.trim.isEmpty) return Array.empty[IndexedToken]

    if (padWithSequenceTokens) {
      originalText = sentencePadding._1 + originalText + sentencePadding._2
    }

    var textList: ListBuffer[String] = ListBuffer(originalText)

    for (specialToken <- specialTokens.allTokens) {
      val currentTokenContent = specialToken.content
      if (currentTokenContent.nonEmpty) { // Avoid issues with empty special tokens
        val regex = java.util.regex.Pattern.quote(currentTokenContent).r
        val newList = ListBuffer[String]()
        for (segment <- textList) {
          if (specialTokens.contains(segment)) {
            // Already identified as a special token in a previous iteration
            newList += segment
          } else {
            // Split this segment by the current special token, preserving the token
            var lastIndex = 0
            regex.findAllMatchIn(segment).foreach { m =>
              if (m.start > lastIndex) {
                newList += segment.substring(lastIndex, m.start)
              }
              newList += m.matched // Add the special token itself
              lastIndex = m.end
            }
            if (lastIndex < segment.length) {
              newList += segment.substring(lastIndex)
            }
          }
        }
        textList = newList.filter(_.nonEmpty) // Update list and remove potential empty strings
      }
    }

    // Adjust index calculations to account for potential padding
    val textWithPadding = originalText // Use the potentially padded text for indexing
    val indexOffsetAdjustment =
      if (padWithSequenceTokens) sentence.start - sentencePadding._1.length else sentence.start

    var currentIndex = 0
    val result = mutable.ArrayBuffer[IndexedToken]()
    for (subText <- textList) {
      // Find the start index based on the text *with* padding if applied
      val subTextIndexInPadded = textWithPadding.indexOf(subText, currentIndex)
      if (subTextIndexInPadded == -1) {
        // This case should ideally not happen if splitting logic is correct, but handle defensively
        // Log warning or error? For now, skip.
        println(
          s"Warning: Could not find segment '$subText' starting from index $currentIndex in text '$textWithPadding'")
        // Attempt to recover by searching from the beginning, though this might be wrong
        currentIndex = textWithPadding.indexOf(subText, 0) + subText.length
      } else {
        currentIndex = subTextIndexInPadded

        // Calculate original index relative to the *unpadded* sentence start
        val originalStartIndex = currentIndex + indexOffsetAdjustment
        val originalEndIndex = originalStartIndex + subText.length - 1

        if (!specialTokens.contains(subText)) {
          // Pass the original start index as the offset for sub-tokenization
          val splitSubText: Array[IndexedToken] = tokenizeSubText(subText, originalStartIndex)
          result.append(splitSubText: _*)
        } else {
          // It's a special token, create IndexedToken directly using original indices
          result.append(IndexedToken(subText, begin = originalStartIndex, end = originalEndIndex))
        }
        // Move currentIndex forward in the padded text
        currentIndex += subText.length
      }
    }
    result.toArray
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
