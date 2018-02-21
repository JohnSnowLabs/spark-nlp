package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.common.{IndexedTaggedWord, TaggedSentence, TaggedWord}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{IntParam, Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.{Map => MMap}
import scala.util.Random

/**
  * Created by Saif Addin on 5/17/2017.
  * Inspired on Averaged Perceptron by Matthew Honnibal
  * https://explosion.ai/blog/part-of-speech-pos-tagger-in-python
  */
class PerceptronApproach(override val uid: String) extends AnnotatorApproach[PerceptronModel] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  import PerceptronApproach._

  override val description: String = "Averaged Perceptron model to tag words part-of-speech"

  val posCol = new Param[String](this, "posCol", "column of Array of POS tags that match tokens")
  val corpus = new ExternalResourceParam(this, "corpus", "POS tags delimited corpus. Needs 'delimiter' in options")
  val nIterations = new IntParam(this, "nIterations", "Number of iterations in training, converges to better accuracy")

  setDefault(corpus, ExternalResource("/anc-pos-corpus/", ReadAs.LINE_BY_LINE, options=Map("delimiter" -> "|")))
  setDefault(nIterations, 5)

  def setPosColumn(value: String): this.type = set(posCol, value)

  def setCorpus(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "PerceptronApproach needs 'delimiter' in options to associate words with tags")
    set(corpus, value)
  }

  def setCorpus(path: String, delimiter: String, readAs: String = "LINE_BY_LINE"): this.type = {
    require(Seq("LINE_BY_LINE", "SPARK_DATASET").contains(readAs.toUpperCase), "readAs needs to be 'LINE_BY_LINE' or 'SPARK_DATASET'")
    set(corpus, ExternalResource(path, readAs, Map("delimiter" -> delimiter)))
  }

  def setNIterations(value: Int): this.type = set(nIterations, value)

  def this() = this(Identifiable.randomUID("POS"))

  override val annotatorType: AnnotatorType = POS

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  /**
    * Finds very frequent tags on a word in training, and marks them as non ambiguous based on tune parameters
    * ToDo: Move such parameters to configuration
    *
    * @param taggedSentences    Takes entire tagged sentences to find frequent tags
    * @param frequencyThreshold How many times at least a tag on a word to be marked as frequent
    * @param ambiguityThreshold How much percentage of total amount of words are covered to be marked as frequent
    */
  private def buildTagBook(
                            taggedSentences: Array[TaggedSentence],
                            frequencyThreshold: Int = 20,
                            ambiguityThreshold: Double = 0.97
                          ): Map[String, String] = {

    val tagFrequenciesByWord = taggedSentences
      .flatMap(_.taggedWords)
      .groupBy(_.word.toLowerCase)
      .mapValues(_.groupBy(_.tag).mapValues(_.length))

    tagFrequenciesByWord.filter { case (_, tagFrequencies) =>
      val (_, mode) = tagFrequencies.maxBy(_._2)
      val n = tagFrequencies.values.sum
      n >= frequencyThreshold && (mode / n.toDouble) >= ambiguityThreshold
    }.map { case (word, tagFrequencies) =>
      val (tag, _) = tagFrequencies.maxBy(_._2)
      logger.debug(s"TRAINING: Ambiguity discarded on: << $word >> set to: << $tag >>")
      (word, tag)
    }
  }

  /**
    * Trains a model based on a provided CORPUS
    *
    * @return A trained averaged model
    */
  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): PerceptronModel = {
    /**
      * Generates TagBook, which holds all the word to tags mapping that are not ambiguous
      */
    val taggedSentences: Array[TaggedSentence] = if (get(posCol).isDefined) {
      import ResourceHelper.spark.implicits._
      val tokenColumn = dataset.schema.fields
        .find(f => f.metadata.contains("annotatorType") && f.metadata.getString("annotatorType") == AnnotatorType.TOKEN)
        .map(_.name).get
      dataset.select(tokenColumn, $(posCol))
        .as[(Array[Annotation], Array[String])]
        .map{
          case (annotations, posTags) =>
            lazy val strTokens = annotations.map(_.result).mkString("#")
            lazy val strPosTags = posTags.mkString("#")
            require(annotations.length == posTags.length, s"Cannot train from $posCol since there" +
              s" is a row with different amount of tags and tokens:\n$strTokens\n$strPosTags")
            TaggedSentence(annotations.zip(posTags)
              .map{case (annotation, posTag) => IndexedTaggedWord(annotation.result, posTag, annotation.start, annotation.end)}
            )
        }.collect
    } else {
      ResourceHelper.parseTupleSentences($(corpus))
    }
    val taggedWordBook = buildTagBook(taggedSentences)
    /** finds all distinct tags and stores them */
    val classes = taggedSentences.flatMap(_.tags).distinct
    val initialModel = new AveragedPerceptron(
      classes,
      taggedWordBook,
      MMap()
    )
    /**
      * Iterates for training
      */
    val trainedModel = (1 to $(nIterations)).foldLeft(initialModel) { (iteratedModel, iteration) => {
      logger.debug(s"TRAINING: Iteration n: $iteration")
      /**
        * In a shuffled sentences list, try to find tag of the word, hold the correct answer
        */
      Random.shuffle(taggedSentences.toList).foldLeft(iteratedModel) { (model, taggedSentence) =>

        /**
          * Defines a sentence context, with room to for look back
          */
        var prev = START(0)
        var prev2 = START(1)
        val context = START ++: taggedSentence.words.map(w => normalized(w)) ++: END
        taggedSentence.words.zipWithIndex.foreach { case (word, i) =>
            val guess = taggedWordBook.getOrElse(word.toLowerCase,{
                /**
                  * if word is not found, collect its features which are used for prediction and predict
                  */
                val features = getFeatures(i, word, context, prev, prev2)
                val guess = model.predict(features)
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
      }
    }}
    trainedModel.averageWeights()
    logger.debug("TRAINING: Finished all iterations")
    new PerceptronModel().setModel(trainedModel)
  }

}
object PerceptronApproach extends DefaultParamsReadable[PerceptronApproach] {

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
                         ): List[(String, Int)] = {
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
    features.toList
  }
}