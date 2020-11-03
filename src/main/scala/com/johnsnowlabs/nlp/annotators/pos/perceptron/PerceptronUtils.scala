package com.johnsnowlabs.nlp.annotators.pos.perceptron

import scala.collection.mutable.{Map => MMap}

trait PerceptronUtils  {

  private[perceptron] val START = Array("-START-", "-START2-")
  private[perceptron] val END = Array("-END-", "-END2-")

  /**
    * Specific normalization rules for this POS Tagger to avoid unnecessary tagging
    * @return
    */
  private[perceptron] def normalized(word: String): String = {
    if (word.contains("-") && word.head != '-') {
      "!HYPEN"
    } else if (word.forall(_.isDigit) && word.length == 4) {
      "!YEAR"
    } else if (word.head.isDigit) {
      "!DIGITS"
    } else {
      word.toLowerCase
    }
  }

  /**
    * Method used when a word tag is not  certain. the word context is explored and features collected
    * @param init word position in a sentence
    * @param word word itself
    * @param context surrounding words of positions -2 and +2
    * @param prev holds previous tag result
    * @param prev2 holds pre previous tag result
    * @return A list of scored features based on how frequently they appear in a context
    */
  private[perceptron] def getFeatures(
                                       init: Int,
                                       word: String,
                                       context: Array[String],
                                       prev: String,
                                       prev2: String
                                     ): Map[String, Int] = {
    val features = MMap[String, Int]().withDefaultValue(0)
    def add(keyName: String): Unit = {
      features(keyName) += 1
    }
    val i = init + START.length
    add("bias")
    add("i suffix" + " " + word.takeRight(3))
    add("i pref1" + " " + word.head)
    add("i-1 tag" + " " + prev)
    add("i-2 tag" + " " + prev2)
    add("i tag+i-2 tag" + " " + prev + " " + prev2)
    add("i word" + " " + context(i))
    add("i-1 tag+i word" + " " + prev + " " + context(i))
    add("i-1 word" + " " + context(i-1))
    add("i-1 suffix" + " " + context(i-1).takeRight(3))
    add("i-2 word" + " " + context(i-2))
    add("i+1 word" + " " + context(i+1))
    add("i+1 suffix" + " " + context(i+1).takeRight(3))
    add("i+2 word" + " " + context(i+2))
    features.toMap
  }
}
