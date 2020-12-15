package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.AnnotatorApproach
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.{Map => MMap}

/** Averaged Perceptron model to tag words part-of-speech.
  *
  * Sets a POS tag to each word within a sentence. Its train data (train_pos) is a spark dataset of POS format values with Annotation columns.
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron]] for further reference on how to use this API.
  *
  * @param uid internal uid required to generate writable annotators
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  * */
class PerceptronApproach(override val uid: String) extends AnnotatorApproach[PerceptronModel]
  with PerceptronTrainingUtils
{

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** veraged Perceptron model to tag words part-of-speech */
  override val description: String = "Averaged Perceptron model to tag words part-of-speech"

  /** column of Array of POS tags that match tokens
    *
    * @group param
    **/
  val posCol = new Param[String](this, "posCol", "column of Array of POS tags that match tokens")
  /** Number of iterations in training, converges to better accuracy
    *
    * @group param
    **/
  val nIterations = new IntParam(this, "nIterations", "Number of iterations in training, converges to better accuracy")
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

  def this() = this(Identifiable.randomUID("POS"))

  /** Output annotator type: POS
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = POS
  /** Input annotator type: TOKEN, DOCUMENT
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)


  /**
    * Trains a model based on a provided CORPUS
    *
    * @return A trained averaged model
    */
  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): PerceptronModel = {

    val taggedSentences = generatesTagBook(dataset)
    val taggedWordBook = buildTagBook(taggedSentences, $(frequencyThreshold), $(ambiguityThreshold))
    /** finds all distinct tags and stores them */
    val classes = taggedSentences.flatMap(_.tags).distinct
    val initialModel = new TrainingPerceptronLegacy(classes, taggedWordBook, MMap())
    val finalModel = trainPerceptron($(nIterations), initialModel, taggedSentences, taggedWordBook)
    logger.debug("TRAINING: Finished all iterations")

    new PerceptronModel().setModel(finalModel)
  }

}

object PerceptronApproach extends DefaultParamsReadable[PerceptronApproach]