package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.annotators.spell.util.Utilities

trait OcrMetrics {

  /*
   * score OCR recognition at a character level, according to,
   * https://csce.ucmss.com/cr/books/2018/LFS/CSREA2018/IPC3481.pdf */
  def score(correct:String, detected:String):Double = {
    val cError = Utilities.levenshteinDistance(correct, detected).toDouble
    val cCorrect = correct.size.toDouble - cError
    val precision = cCorrect / detected.length
    val recall = cCorrect / correct.length
    // fscore
    (2 * precision * recall)/(precision + recall)
  }
}
