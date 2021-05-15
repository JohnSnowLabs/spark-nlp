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
                        padWithSentenceTokens: Boolean = false,
                      ) extends BpeTokenizer(merges, vocab, specialTokens) {

  /**
    * Mapping for bytes to a different set of unicode characters (especially white spaces).
    * This improved model performance for gpt-2. TODO: Only for gpt-2 type bpe
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
  private val encodeByte =
    (tok: String) => tok.foldLeft("")(_ + bytesToUnicodeMapping(_))

  /**
    * Special tokens of the model for processing
    */
  //  override val specialTokens: SpecialTokens = {
  //    val bpeSpecialTokens = new BpeSpecialTokens("roberta")
  //    bpeSpecialTokens.getSpecialTokens
  //  }
  val sentencePadding: (String, String) = (specialTokens.sentenceStart.content, specialTokens.sentenceEnd.content)

  /**
    * split pattern based on gpt2's bpe tokenizer
    */
  private def splitOnPattern(text: String, indexOffset: Int): Array[IndexedToken] = {
    val splitPattern: Regex = raw"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+".r
    splitPattern
      .findAllMatchIn(text)
      .map(tok => IndexedToken(tok.matched, tok.start + indexOffset, tok.end + indexOffset - 1)) // TODO Expected -1?
      .toArray
  }

  /**
    * Tokenize considering special tokens and split pattern
    */
  override def tokenize(
                         sentence: Sentence
                       ): Array[IndexedToken] = {
    var text = sentence.content
    if (text.trim.isEmpty) Array[IndexedToken]()
    else {
      val splitTexts: ListBuffer[String] = ListBuffer()
      var textList: ListBuffer[String] = ListBuffer(text)

      for (transformations <- specialTokens.allTokens) {
        splitTexts.clear()
        for (subText <- textList) {
          if (!specialTokens.contains(subText))
            splitTexts ++= splitOnSpecialToken(transformations, subText)
          else
            splitTexts += subText
        }
        textList = splitTexts.clone()
      }
      if (padWithSentenceTokens) {
        text = sentencePadding._1 + text + sentencePadding._2
        splitTexts.prepend(sentencePadding._1)
        splitTexts.append(sentencePadding._2)
      }
      var currentIndex = 0
      val result = mutable.ArrayBuffer[IndexedToken]()
      for (subText <- splitTexts) {
        val subTextIndex = sentence.start + text.indexOf(subText, currentIndex)
        if (!specialTokens.contains(subText)) {
          val splitSubText = splitOnPattern(subText, subTextIndex)
          result.append(splitSubText: _*)
        } else // subtext is just the special token
          result.append(
            IndexedToken(
              subText,
              begin = subTextIndex,
              end = subTextIndex + subText.length - 1
            )
          )
        currentIndex = subTextIndex + subText.length
      }
      result.toArray
    }
  }

  override def encode(indToken: IndexedToken): Array[TokenPiece] = {
    if (!specialTokens.contains(indToken.token))
      bpe(indToken, encodeByte)
    else
      Array(
        TokenPiece(
          indToken.token,
          indToken.token,
          vocab(indToken.token),
          isWordStart = true,
          indToken.begin,
          indToken.end
        )
      )
  }
}
