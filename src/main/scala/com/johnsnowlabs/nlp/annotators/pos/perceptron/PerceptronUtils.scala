package com.johnsnowlabs.nlp.annotators.pos.perceptron

import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.{Map => MMap}

trait PerceptronUtils  {

  private[perceptron] val START = Array("-START-", "-START2-")
  private[perceptron] val END = Array("-END-", "-END2-")

  private[perceptron] val logger: Logger = LoggerFactory.getLogger("PerceptronTraining")

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
    def add(name: String, args: Array[String] = Array()): Unit = {
      features((name +: args).mkString(" ")) += 1
    }
    val i = init + START.length
    add("bias")
    add("i suffix", Array(word.takeRight(3)))
    add("i pref1", Array(word.head.toString))
    add("i-1 tag", Array(prev))
    add("i-2 tag", Array(prev2))
    add("i tag+i-2 tag", Array(prev, prev2))
    add("i word", Array(context(i)))
    add("i-1 tag+i word", Array(prev, context(i)))
    add("i-1 word", Array(context(i-1)))
    add("i-1 suffix", Array(context(i-1).takeRight(3)))
    add("i-2 word", Array(context(i-2)))
    add("i+1 word", Array(context(i+1)))
    add("i+1 suffix", Array(context(i+1).takeRight(3)))
    add("i+2 word", Array(context(i+2)))
    features.toMap
  }
}
