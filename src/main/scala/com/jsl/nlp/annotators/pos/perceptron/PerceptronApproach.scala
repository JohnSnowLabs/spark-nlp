package com.jsl.nlp.annotators.pos.perceptron

import com.jsl.nlp.annotators.pos.POSApproach

import scala.collection.mutable.{Map => MMap}
import scala.util.Random

/**
  * Created by Saif Addin on 5/17/2017.
  * Inspired on Averaged Perceptron by Matthew Honnibal
  * https://explosion.ai/blog/part-of-speech-pos-tagger-in-python
  */
class PerceptronApproach(trainedModel: AveragedPerceptron) extends POSApproach {

  import PerceptronApproach._

  override val description: String = "Averaged Perceptron tagger, iterative average weights upon training"

  /**
    * Bundles a sentence within context and then finds unambiguous word or predict it
    * @return
    */

  override val model: AveragedPerceptron = trainedModel

  override def tag(rawSentences: Array[String]): Array[TaggedWord] = {
    val sentences = rawSentences.map(Sentence)
    var prev = START(0)
    var prev2 = START(1)
    sentences.map(_.tokenize).flatMap{words => {
      val context: Array[String] = START ++: words.map(_.normalized) ++: END
      words.zipWithIndex.map{case (word, i) =>
        val tag = model.taggedWordBook.find(_.word == word).map(_.tag).getOrElse(
          {
            val features = getFeatures(i, word, context, prev, prev2)
            model.predict(features.toMap)
          }
        )
        prev2 = prev
        prev = tag
        (word, tag)
      }
    }}.map(t => TaggedWord(t._1, t._2))
  }

}
object PerceptronApproach {

  private val START = Array("-START-", "-START2-")
  private val END = Array("-END-", "-END2-")

  implicit def word2str(word: Word): String = word.word
  implicit def sentence2str(sentence: Sentence): Array[Word] = sentence.tokenize


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
    * Supposed to find very frequent tags and record them
    * @param taggedSentences
    */
  private def buildTagBook(
                            taggedSentences: List[TaggedSentence],
                            frequencyThreshold: Int = 20,
                            ambiguityThreshold: Double = 0.97
                          ): List[TaggedWord] = {
    /**
      * This creates counts, a map of words that refer to all possible tags and how many times they appear
      * It holds how many times a word-tag combination appears in the training corpus
      * It is also used in the rest of the tagging process to hold tags
      * It also stores the tag in classes which holds tags
      * Then Find the most frequent tag and its count
      * If there is a very frequent tag, map the word to such tag to disambiguate
      */

    val tagFrequenciesByWord = taggedSentences
      .flatMap(_.tagged)
      .groupBy(_.word)
      .mapValues(_.groupBy(_.tag).mapValues(_.length))

    tagFrequenciesByWord.filter{case (_, tagFrequencies) =>
        val (_, mode) = tagFrequencies.maxBy(_._2)
        val n = tagFrequencies.values.sum
        n >= frequencyThreshold && (mode / n.toDouble) >= ambiguityThreshold
      }.map{case (word, tagFrequencies) =>
        val (tag, _) = tagFrequencies.maxBy(_._2)
        TaggedWord(word, tag)
      }.toList
  }

  def train(taggedSentences: List[TaggedSentence], nIterations: Int = 5): PerceptronApproach = {
    /**
      * Generates tagdict, which holds all the word to tags mapping
      * Adds the found tags to the tags available in the model
      */
    val taggedWordBook = buildTagBook(taggedSentences)
    val classes = taggedSentences.flatMap(_.tags).distinct
    val initialModel = new AveragedPerceptron(taggedWordBook, classes, MMap())
    /**
      * Iterates for training
      */
    val trainedModel = (1 to nIterations).foldRight(initialModel){(_, iteratedModel) => {
      /**
        * Defines a sentence context, with room to for look back
        */
      var prev = START(0)
      var prev2 = START(1)
      /**
        * In a shuffled sentences list, try to find tag of the word, hold the correct answer
        */
      Random.shuffle(taggedSentences).foldRight(iteratedModel)
      {(taggedSentence, model) =>
        val context = START ++: taggedSentence.words.map(_.normalized) ++: END
        taggedSentence.words.zipWithIndex.foreach{case (word, i) =>
          val guess = model.taggedWordBook.find(_.word == word).map(_.tag).getOrElse({
            /**
              * if word is not found, collect its features which are used for prediction and predict
              */
            val features = getFeatures(i, word, context, prev, prev2)
            val guess = model.predict(features.toMap)
            /**
              * Update the model based on the prediction results
              */
            model.update(taggedSentence.tags(i), guess, features.toMap)
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
        model
      }.averagedModel
    }}
    new PerceptronApproach(trainedModel)
  }
}