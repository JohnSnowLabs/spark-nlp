package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.common.{IndexedTaggedWord, TaggedSentence}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.{Map => MMap}
import scala.util.Random

/**
  * Created by Saif Addin on 5/17/2017.
  * Inspired on Averaged Perceptron by Matthew Honnibal
  * https://explosion.ai/blog/part-of-speech-pos-tagger-in-python
  */
class PerceptronApproach(override val uid: String) extends AnnotatorApproach[PerceptronModel] with PerceptronUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val description: String = "Averaged Perceptron model to tag words part-of-speech"

  val posCol = new Param[String](this, "posCol", "column of Array of POS tags that match tokens")
  val corpus = new ExternalResourceParam(this, "corpus", "POS tags delimited corpus. Needs 'delimiter' in options")
  val nIterations = new IntParam(this, "nIterations", "Number of iterations in training, converges to better accuracy")

  setDefault(nIterations, 5)

  def setPosColumn(value: String): this.type = set(posCol, value)

  def setCorpus(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "PerceptronApproach needs 'delimiter' in options to associate words with tags")
    set(corpus, value)
  }

  def setCorpus(path: String,
                delimiter: String,
                readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                options: Map[String, String] = Map("format" -> "text")): this.type =
    set(corpus, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))

  def setNIterations(value: Int): this.type = set(nIterations, value)

  def this() = this(Identifiable.randomUID("POS"))

  override val outputAnnotatorType: AnnotatorType = POS

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

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
      val dataFrameSchemaField = dataset.schema.fields(0).metadata

      require(dataFrameSchemaField.contains("annotatorType"), s"Cannot train from DataFrame without POS annotatorType")
      require(dataFrameSchemaField.getString("annotatorType")== AnnotatorType.POS, s"Cannot train from DataFrame without POS annotatorType")
      val posColumn = dataset.schema.fields
        .find(f => f.metadata.contains("annotatorType") && f.metadata.getString("annotatorType") == AnnotatorType.POS)
        .map(_.name).get

      dataset.select(posColumn)
        .as[Array[Annotation]]
        .map{
          annotations =>
            TaggedSentence(annotations
              .map{annotation => IndexedTaggedWord(annotation.metadata("word"), annotation.result, annotation.begin, annotation.end)}
            )
        }.collect
    } else {
      ResourceHelper.parseTupleSentences($(corpus))
    }
    val taggedWordBook = buildTagBook(taggedSentences)
    /** finds all distinct tags and stores them */
    val classes = taggedSentences.flatMap(_.tags).distinct
    val initialModel = new TrainingPerceptronLegacy(
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
            model.update(taggedSentence.tags(i), guess, features)
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
    val finalModel = trainedModel.averageWeights()
    logger.debug("TRAINING: Finished all iterations")
    new PerceptronModel().setModel(finalModel)
  }
}

object PerceptronApproach extends DefaultParamsReadable[PerceptronApproachDistributed]