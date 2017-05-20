package com.jsl.nlp.annotators.pos.perceptron

import com.jsl.nlp.annotators.pos.{POSApproach, TaggedWord}

import scala.collection.mutable.{ArrayBuffer, Map => MMap, Set => MSet}
import scala.util.Random

/**
  * Created by Saif Addin on 5/17/2017.
  * Inspired on Averaged Perceptron by Matthew Honnibal
  * https://explosion.ai/blog/part-of-speech-pos-tagger-in-python
  */
class PerceptronApproach extends POSApproach {

  override val description: String = "Averaged Perceptron tagger, iterative average weights upon training"

  private val tokenRegex = "\\W+"

  /**
    * Very important object that holds the list of all tags
    */
  private val classes: MSet[String] = MSet()
  /**
    * Very important object for certain word-tag
    */
  private val tagdict: MMap[String, String] = MMap()

  private val START = Array("-START-", "-START2-")
  private val END = Array("-END-", "-END2-")

  val model = new AveragedPerceptron()

  /**
    * ToDo: Analyze whether we can re-use any tokenizer from annotators
    * @return
    */
  private def tokenize(sentences: Array[String]): Array[Array[String]] = {
    sentences.map(sentence => sentence.split(tokenRegex))
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

  /**
    * Method used when a word tag is not  certain. the word context is explored and features collected
    * @param init
    * @param word
    * @param context
    * @param prev holds previous tag
    * @param prev2 holds previous tag
    * @return
    */
  private def getFeatures(
                           init: Int,
                           word: String,
                           context: Array[String],
                           prev: String,
                           prev2: String
                         ): MMap[String, Int] = {
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
    features
  }

  /**
    * Bundles a sentence within context and then finds unambiguous word or predict it
    * @param sentences
    * @return
    */
  override def tag(sentences: Array[String]): Array[TaggedWord] = {
    var prev = START(0)
    var prev2 = START(1)
    val tokens = ArrayBuffer[(String, String)]()
    tokenize(sentences).foreach{words => {
      val context = START ++: words.map(dataPreProcess) ++: END
      words.zipWithIndex.foreach{case (word, i) =>
        val tag = tagdict.getOrElse(
          word,
          {
            val features = getFeatures(i, word, context, prev, prev2)
            model.predict(features.toMap)
          }
        )
        tokens.append((word, tag))
        prev2 = prev
        prev = tag
      }
    }}
    tokens.toArray.map(t => TaggedWord(t._1, t._2))
  }

  /**
    * Supposed to find very frequent tags and record them
    * @param sentences
    */
  private def makeTagDict(sentences: List[(List[String], List[String])]): Unit = {
    /**
      * This creates counts, a map of words that refer to all possible tags and how many times they appear
      * It holds how many times a word-tag combination appears in the training corpus
      * It is also used in the rest of the tagging process to hold tags
      * It also stores the tag in classes which holds tags
      */
    val counts: MMap[String, MMap[String, Int]] = MMap()
    sentences.foreach{case (words, tags) =>
      words.zip(tags).foreach{case (word, tag) =>
        counts.getOrElseUpdate(word, MMap().withDefaultValue(0))
        counts(word)(tag) += 1
        classes.add(tag)
      }
    }
    /**
      * This part Find the most frequent tag and its count
      * If there is a very frequent tag, map the word to such tag to disambiguate
      */
    val freqThreshold = 20
    val ambiguityThreshold = 0.97
    counts.foreach{case (word, tagFreqs) =>
      val (tag, mode) = tagFreqs.maxBy(_._2)
      val n = tagFreqs.values.sum
      if (n >= freqThreshold && (mode / n.toDouble) >= ambiguityThreshold) {
        tagdict(word) = tag
      }
    }
  }
  def train(sentences: List[(List[String], List[String])], nIterations: Int = 5): Unit = {
    /**
      * Generates tagdict, which holds all the word to tags mapping
      * Adds the found tags to the tags available in the model
      */
    makeTagDict(sentences)
    model.classes = classes.toSet
    /**
      * Defines a sentence context, with room to for lookback
      */
    var prev = START(0)
    var prev2 = START(1)
    /**
      * Iterates for training
      */
    (1 to nIterations).foreach{_ => {
      /**
        * In a shuffled sentences list, try to find tag of the word, hold the correct answer
        */
      Random.shuffle(sentences).foreach{case (words, tags) =>
        val context = START ++: words.map(dataPreProcess) ++: END
        words.zipWithIndex.foreach{case (word, i) =>
          val guess = tagdict.getOrElse(word, {
            /**
              * if word is not found, collect its features which are used for prediction and predict
              */
            val features = getFeatures(i, word, context, prev, prev2)
            val guess = model.predict(features.toMap)
            /**
              * Update the model based on the prediction results
              */
            model.update(tags(i), guess, features.toMap)
            /**
              * return the guess
              */
            guess
          })
          /**
            * shift the context
            */
          prev2 = prev
          prev = guess
        }
      }
    }}
    /**
      *
      */
    model.averageWeights()
  }

}
