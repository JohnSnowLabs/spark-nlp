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

package com.johnsnowlabs.nlp.annotators.keyword.yake.util

import org.slf4j.LoggerFactory

object Utilities {
  private val logger = LoggerFactory.getLogger("YakeUtilities")

  /**
    * Given a word, assign a tag based on the content
    * Tags
    * d - digit
    * u - unparsable
    * a - Acronym
    * n - Noun
    * p - parsable
    * @param word Input word that needs to be tagged
    * @param i Sentence index
    * @return
    */
  def getTag(word: String, i: Int): String = {
    try {
      val word2 = word.replace(",", "")
      word2.toFloat
      "d"
    } catch {
      case x: Exception => {
        val cdigit = word.count(_.isDigit)
        val calpha = word.count(_.isLetter)
        if ((cdigit > 0 && calpha > 0) || (calpha == 0 && cdigit == 0) || (cdigit+calpha != word.length)) {
          "u"
        }
        else if (word.length() == word.count(_.isUpper)) {
          "a"
        }
        else if (word.count(_.isUpper) == 1 && word.length() > 1 && word.charAt(0).isUpper && i > 0) {
          "n"
        }
        else {
          "p"
        }
      }
    }
  }

  def medianCalculator(seq: Seq[Int]): Int = {
    //In order if you are not sure that 'seq' is sorted
    val sortedSeq = seq.sortWith(_ < _)

    if (seq.size % 2 == 1) sortedSeq(sortedSeq.size / 2)
    else {
      val (up, down) = sortedSeq.splitAt(seq.size / 2)
      (up.last + down.head) / 2
    }
  }
}