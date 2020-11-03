package com.johnsnowlabs.nlp.annotators.ws

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronTrainingUtils, TrainingPerceptronLegacy}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.{Map => MMap}

class WordSegmenterApproach(override val uid: String) extends AnnotatorApproach[WordSegmenterModel]
  with PerceptronTrainingUtils
{

  def this() = this(Identifiable.randomUID("WORD_SEGMENTER"))

  override val description: String = "Word segmentation"

  /** column of Array of POS tags that match tokens
   *
   * @group param
   **/
  val posCol = new Param[String](this, "posCol", "column of Array of POS tags that match tokens")
  /** Number of iterations in training, converges to better accuracy
   *
   * @group param
   **/
  val nIterations = new IntParam(this, "nIterations",
    "Number of iterations in training, converges to better accuracy")
  setDefault(nIterations, 5)

  /** How many times at least a tag on a word to be marked as frequent
   *
   * @group param
   **/
  val frequencyThreshold = new IntParam(this, "frequencyThreshold",
    "How many times at least a tag on a word to be marked as frequent")
  setDefault(frequencyThreshold, 20)

  /** How much percentage of total amount of words are covered to be marked as frequent
   *
   * @group param
   **/
  val ambiguityThreshold = new DoubleParam(this, "ambiguityThreshold",
    "How much percentage of total amount of words are covered to be marked as frequent")
  setDefault(ambiguityThreshold, 0.97)

  /** Column containing an array of POS Tags matching every token on the line.
   *
   * @group setParam
   **/
  def setPosColumn(value: String): this.type = set(posCol, value)

  /** Number of iterations for training. May improve accuracy but takes longer. Default 5.
   *
   * @group setParam
   **/
  def setNIterations(value: Int): this.type = set(nIterations, value)

  def setFrequencyThreshold(value: Int): this.type = set(frequencyThreshold, value)

  def setAmbiguityThreshold(value: Double): this.type = set(ambiguityThreshold, value)

  /** Number of iterations for training. May improve accuracy but takes longer. Default 5.
   *
   * @group getParam
   **/
  def getNIterations: Int = $(nIterations)


  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): WordSegmenterModel = {
    val taggedSentences = generatesTagBook(dataset)
    val taggedWordBook = buildTagBook(taggedSentences, $(frequencyThreshold), $(ambiguityThreshold))
    /** finds all distinct tags and stores them */
    val classes = taggedSentences.flatMap(_.tags).distinct
    val initialModel = new TrainingPerceptronLegacy(classes, taggedWordBook, MMap())
    val finalModel = trainPerceptron($(nIterations), initialModel, taggedSentences, taggedWordBook)

    new WordSegmenterModel()
      .setModel(finalModel)
  }

  override val outputAnnotatorType: AnnotatorType = TOKEN
  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)
}

object WordSegmenterApproach extends DefaultParamsReadable[WordSegmenterApproach]