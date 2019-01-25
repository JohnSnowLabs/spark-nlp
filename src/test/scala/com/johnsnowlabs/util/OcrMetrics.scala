package com.johnsnowlabs.util
import com.johnsnowlabs.nlp.annotators.spell.common.LevenshteinDistance

trait OcrMetrics extends LevenshteinDistance {

  /*
   * score OCR recognition at a character level, according to,
   * https://csce.ucmss.com/cr/books/2018/LFS/CSREA2018/IPC3481.pdf */
  def score(correct:String, detected:String):Double = {
    val cError = levenshteinDistance(correct, detected).toDouble
    val cCorrect = correct.size.toDouble
    val precision = cCorrect / detected.length
    val recall = cCorrect / correct.length
    // fscore
    (2 * precision * recall)/(precision + recall)
  }
}
