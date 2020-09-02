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