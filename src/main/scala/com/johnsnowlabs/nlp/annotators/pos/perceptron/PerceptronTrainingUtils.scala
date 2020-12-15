package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.common.{IndexedTaggedWord, TaggedSentence}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.sql.Dataset
import org.slf4j.{Logger, LoggerFactory}

import scala.util.Random

trait PerceptronTrainingUtils extends PerceptronUtils {

  private[perceptron] val logger: Logger = LoggerFactory.getLogger("PerceptronApproachUtils")

  /**
   * Generates TagBook, which holds all the word to tags mapping that are not ambiguous
   */
  def generatesTagBook(dataset: Dataset[_]): Array[TaggedSentence] = {
    val taggedSentences = {
      import ResourceHelper.spark.implicits._

      val datasetSchemaFields = dataset.schema.fields
        .find(f => f.metadata.contains("annotatorType") && f.metadata.getString("annotatorType") == AnnotatorType.POS)

      require(datasetSchemaFields.map(_.name).isDefined, s"Cannot train from DataFrame without POS annotatorType by posCol")

      val posColumn = datasetSchemaFields.map(_.name).get

      dataset.select(posColumn)
        .as[Array[Annotation]]
        .map{
          annotations =>
            TaggedSentence(annotations
              .map{annotation => IndexedTaggedWord(annotation.metadata("word"), annotation.result, annotation.begin, annotation.end)}
            )
        }.collect
    }
    taggedSentences
  }

  /**
   * Finds very frequent tags on a word in training, and marks them as non ambiguous based on tune parameters
   * ToDo: Move such parameters to configuration
   *
   * @param taggedSentences    Takes entire tagged sentences to find frequent tags
   * @param frequencyThreshold How many times at least a tag on a word to be marked as frequent
   * @param ambiguityThreshold How much percentage of total amount of words are covered to be marked as frequent
   */
  def buildTagBook(taggedSentences: Array[TaggedSentence], frequencyThreshold: Int, ambiguityThreshold: Double):
    Map[String, String] =
  {

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
   * Iterates for training
   */
  def trainPerceptron(nIterations: Int, initialModel: TrainingPerceptronLegacy,
                      taggedSentences: Array[TaggedSentence], taggedWordBook: Map[String, String]):
  AveragedPerceptron =
  {
    val trainedModel = (1 to nIterations).foldLeft(initialModel) { (iteratedModel, iteration) => {
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
    trainedModel.averageWeights()
  }

}
