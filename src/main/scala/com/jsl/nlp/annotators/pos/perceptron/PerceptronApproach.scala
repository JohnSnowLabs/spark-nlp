package com.jsl.nlp.annotators.pos.perceptron

import com.jsl.nlp.annotators.pos.{POSApproach, TaggedWord}

import scala.collection.mutable.{ArrayBuffer, Map => MMap}

/**
  * Created by Saif Addin on 5/17/2017.
  * Inspired on Averaged Perceptron by Matthew Honnibal
  * https://explosion.ai/blog/part-of-speech-pos-tagger-in-python
  */
class PerceptronApproach extends POSApproach {

  override val description: String = "Averaged Perceptron tagger, iterative average weights upon training"

  private val tokenRegex = "\\w".r

  private val tagdict: MMap[String, String] = MMap()

  private val START = Array("-START-", "-START2-")
  private val END = Array("-END-", "-END2-")

  val model = new AveragedPerceptron()

  /**
    * ToDo: Analyze whether we can re-use any tokenizer from annotators
    * @return
    */
  private def tokenize(sentences: Array[String]): Array[Array[String]] = {
    sentences.map(sentence => tokenRegex.findAllMatchIn(sentence).map(_.matched).toArray)
  }

  private def dataPreProcess(word: String): String = {
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

  private def getFeatures(
                           init: Int,
                           word: String,
                           context: Array[String],
                           prev: String,
                           prev2: String
                         ): MMap[String, Int] = {
    val features = MMap[String, Int]().withDefault(_ => 0)
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
    features
  }

  override def tag(sentences: Array[String]): Array[TaggedWord] = {
    var prev = START(0)
    var prev2 = START(1)
    val tokens = ArrayBuffer[(String, String)]()
    tokenize(sentences).foreach{words => {
      val context = START ++: words.map(w => dataPreProcess(w)) ++: END
      words.zipWithIndex.foreach{case (word, i) => {
        val tag = tagdict.getOrElse(
          word,
          {
            val features = getFeatures(i, word, context, prev, prev2)
            //ToDo: Careful with Mutable to Immutable
            model.predict(features.toMap)
          }
        )
        tokens.append((word, tag))
        prev2 = prev
        prev = tag
      }}
    }}
    tokens.toArray.map(t => TaggedWord(t._1, t._2))
  }

}
