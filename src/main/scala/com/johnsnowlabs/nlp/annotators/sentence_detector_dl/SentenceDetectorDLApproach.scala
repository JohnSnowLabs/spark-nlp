package com.johnsnowlabs.nlp.annotators.sentence_detector_dl


import com.johnsnowlabs.ml.tensorflow.{TensorflowSentenceDetectorDL, TensorflowWrapper, Variables}
import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.commons.io.IOUtils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{FloatParam, IntParam, Param, StringArrayParam, BooleanParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.col
import org.tensorflow.Graph
import org.tensorflow.proto.framework.GraphDef

import scala.collection.mutable.WrappedArray
import scala.io.Source

class SentenceDetectorDLApproach(override val uid: String)
  extends AnnotatorApproach[SentenceDetectorDLModel] {

  def this() = this(Identifiable.randomUID("SentenceDetectorDLApproach"))

  /** Trains TensorFlow model for multi-class text classification */
  override val description = "Trains a deep sentence detector"


  /** Input annotator type : SENTENCE_EMBEDDINGS
   *
   * @group anno
   **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Output annotator type : CATEGORY
   *
   * @group anno
   **/
  override val outputAnnotatorType: String = DOCUMENT

  /** Model architecture
   *
   * @group param
   **/
  var modelArchitecture = new Param[String](this, "modelArchitecture", "Model Architecture: one of (CNN)")


  /** Set architecture
   *
   * @group setParam
   **/
  def setModel(modelArchitecture: String): SentenceDetectorDLApproach.this.type = set(this.modelArchitecture, modelArchitecture)

  /** Get model architecture
   *
   * @group getParam
   **/
  def getModel: String = $(this.modelArchitecture)


  /** Impossible penultimates
   *
   * @group param
   **/
  val impossiblePenultimates = new StringArrayParam(this, "impossiblePenultimates", "Impossible penultimates")

  /** Set impossible penultimates
   *
   * @group setParam
   **/
  def setImpossiblePenultimates(impossiblePenultimates: Array[String]):
    SentenceDetectorDLApproach.this.type = set(this.impossiblePenultimates, impossiblePenultimates)

  /** Get impossible penultimates
   *
   * @group getParam
   **/
  def getImpossiblePenultimates: Array[String] = $(this.impossiblePenultimates)


  /** Maximum number of epochs to train
   *
   * @group param
   **/
  val epochsNumber = new IntParam(this, "epochsNumber", "Number of epochs to train")

  /** Maximum number of epochs to train
   *
   * @group setParam
   **/
  def setEpochsNumber(epochs: Int): SentenceDetectorDLApproach.this.type = set(this.epochsNumber, epochs)


  /** Maximum number of epochs to train
   *
   * @group getParam
   **/
  def getEpochsNumber: Int = $(this.epochsNumber)

  /** Fix imbalance in training set
   *
    */
  /** Path to folder to output logs. If no path is specified, no logs are generated
   *
   * @group param
   **/
  val outputLogsPath = new Param[String](this, "outputLogsPath", "Folder path to save training logs")

  /** Set the output log path
   *
   * @group setParam
   **/
  def setOutputLogsPath(outputLogsPath: String): SentenceDetectorDLApproach.this.type = set(this.outputLogsPath, outputLogsPath)

  /** Get output logs path
   *
   * @group getParam
   **/
  def getOutputLogsPath: String = $(this.outputLogsPath)

  /** Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
   *
   * @group param
   **/
  val validationSplit = new FloatParam(this, "validationSplit", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.")

  /** Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
   *
   * @group setParam
   **/
  def setValidationSplit(validationSplit: Float): SentenceDetectorDLApproach.this.type = set(this.validationSplit, validationSplit)

  /** Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
   *
   * @group getParam
   **/
  def getValidationSplit: Float = $(this.validationSplit)

  private var graphFilename: Option[String] = None

  def getGraphFilename: String = {
    graphFilename.getOrElse("sentence_detector_dl/%s.pb".format(getModel))
  }

  def setGraphFile(graphFilename: String): SentenceDetectorDLApproach.this.type = {
    this.graphFilename = Some(graphFilename)
    this
  }

  /** A flag indicating whether to split sentences into different Dataset rows. Useful for higher parallelism in
    * fat rows. Defaults to false.
    *
    * @group getParam
    **/
  def explodeSentences = new BooleanParam(this, "explodeSentences", "Split sentences in separate rows")


  /** Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat rows. Defaults to false.
    *
    * @group setParam
    **/
  def setExplodeSentences(value: Boolean): SentenceDetectorDLApproach.this.type = set(this.explodeSentences, value)


  /** Whether to split sentences into different Dataset rows. Useful for higher parallelism in fat rows. Defaults to false.
    *
    * @group getParam
    **/
  def getExplodeSentences: Boolean = $(this.explodeSentences)

  setDefault(
    modelArchitecture -> "cnn",
    impossiblePenultimates -> Array(
      "Bros", "No", "al", "vs", "etc", "Fig", "Dr", "Prof", "PhD", "MD", "Co", "Corp", "Inc",
      "bros", "VS", "Vs", "ETC", "fig", "dr", "prof", "PHD", "phd", "md", "co", "corp", "inc",
      "Jan", "Feb", "Mar", "Apr", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
      "St", "st",
      "AM", "PM", "am", "pm",
      "e.g", "f.e", "i.e"
    ),
    epochsNumber -> 5,
    outputLogsPath -> "",
    validationSplit -> 0.0f,
    explodeSentences -> false
  )

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): SentenceDetectorDLModel = {

    var text = dataset
      .select(getInputCols.map(col): _*)
      .collect()
      .map(row => {
        (0 until getInputCols.length).map(
          i =>
            row
            .get(i)
              .asInstanceOf[WrappedArray[Row]]
              .map(s => Annotation(s).result).mkString("\n")
        ).mkString("\n")
      })
      .mkString("\n")


    val encoder = new SentenceDetectorDLEncoder()
    encoder.buildVocabulary(text)

    val data = encoder.getTrainingData(text)

    val graph = new Graph()
    val graphStream = ResourceHelper.getResourceStream(getGraphFilename)
    val graphBytesDef = IOUtils.toByteArray(graphStream)
    graph.importGraphDef(GraphDef.parseFrom(graphBytesDef))

    val tfWrapper = new TensorflowWrapper(
      Variables(Array.empty[Byte], Array.empty[Byte]),
      graphBytesDef
    )

    /**  FIXME inspect ops for init */
    //graph.operations().foreach(println)

    val tfModel = new TensorflowSentenceDetectorDL(
      tfWrapper,
      outputLogsPath = if (getOutputLogsPath.length > 0) Some(getOutputLogsPath) else None)

    tfModel.train(
      data._2,
      data._1.map(Array(_)),
      batchSize = 32,
      epochsNumber = getEpochsNumber,
      learningRate = 0.0001f,
      validationSplit = getValidationSplit,
      classWeights = Array(1.0f),
      dropout = 1.0f,
      uuid = "sentence_detector_dl"
    )

    val model = new SentenceDetectorDLModel

    model.setModel(getModel)
    model.setEncoder(encoder)
    model.setImpossiblePenultimates(getImpossiblePenultimates)
    model.setupTFClassifier(dataset.sparkSession, tfModel.model)

    model
  }
}
