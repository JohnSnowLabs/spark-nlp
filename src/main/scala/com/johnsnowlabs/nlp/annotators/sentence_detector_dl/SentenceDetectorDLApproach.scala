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

/**
  * Trains an annotator that detects sentence boundaries using a deep learning approach.
  *
  * For pretrained models see [[SentenceDetectorDLModel]].
  *
  * Currently, only the CNN model is supported for training, but in the future the architecture of the model can
  * be set with `setModelArchitecture`.
  *
  * The default model `"cnn"` is based on the paper
  * [[https://konvens.org/proceedings/2019/papers/KONVENS2019_paper_41.pdf Deep-EOS: General-Purpose Neural Networks for Sentence Boundary Detection (2020, Stefan Schweter, Sajawel Ahmed)]]
  * using a CNN architecture. We also modified the original implementation a little bit to cover broken sentences and some impossible end of line chars.
  *
  * Each extracted sentence can be returned in an Array or exploded to separate rows,
  * if `explodeSentences` is set to `true`.
  *
  * For extended examples of usage, see the [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sentence_detector_dl/SentenceDetectorDLSpec.scala SentenceDetectorDLSpec]].
  *
  * ==Example==
  * The training process needs data, where each data point is a sentence.
  *
  * In this example the `train.txt` file has the form of
  * {{{
  * ...
  * Slightly more moderate language would make our present situation – namely the lack of progress – a little easier.
  * His political successors now have great responsibilities to history and to the heritage of values bequeathed to them by Nelson Mandela.
  * ...
  * }}}
  * where each line is one sentence.
  * Training can then be started like so:
  * {{{
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLApproach
  * import org.apache.spark.ml.Pipeline
  *
  * val trainingData = spark.read.text("train.txt").toDF("text")
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentenceDetector = new SentenceDetectorDLApproach()
  *   .setInputCols(Array("document"))
  *   .setOutputCol("sentences")
  *   .setEpochsNumber(100)
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector))
  *
  * val model = pipeline.fit(trainingData)
  * }}}
  * @see [[SentenceDetectorDLModel]] for pretrained models
  * @see [[com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector SentenceDetector]] for non deep learning extraction
  * @param uid required uid for storing annotator to disk
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
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class SentenceDetectorDLApproach(override val uid: String)
  extends AnnotatorApproach[SentenceDetectorDLModel] {

  def this() = this(Identifiable.randomUID("SentenceDetectorDLApproach"))

  /** Trains TensorFlow model for multi-class text classification */
  override val description = "Trains a deep sentence detector"


  /** Input annotator type : DOCUMENT
   *
   * @group anno
   **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Output annotator type : DOCUMENT
   *
   * @group anno
   **/
  override val outputAnnotatorType: String = DOCUMENT

  /** Model architecture (Default: `"cnn"`)
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


  /** Impossible penultimates, which should not be split on
    * Default:
    * {{{
    * Array(
    *   "Bros", "No", "al", "vs", "etc", "Fig", "Dr", "Prof", "PhD", "MD", "Co", "Corp", "Inc",
    *   "bros", "VS", "Vs", "ETC", "fig", "dr", "prof", "PHD", "phd", "md", "co", "corp", "inc",
    *   "Jan", "Feb", "Mar", "Apr", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
    *   "St", "st",
    *   "AM", "PM", "am", "pm",
    *   "e.g", "f.e", "i.e"
    * )
    * }}}
    *
    * @group param
    * */
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


  /** Maximum number of epochs to train (Default: `5`)
   *
   * @group param
   **/
  val epochsNumber = new IntParam(this, "epochsNumber", "Number of epochs to train")

  /** Maximum number of epochs to train (Default: `5`)
   *
   * @group setParam
   **/
  def setEpochsNumber(epochs: Int): SentenceDetectorDLApproach.this.type = set(this.epochsNumber, epochs)


  /** Maximum number of epochs to train (Default: `5`)
   *
   * @group getParam
   **/
  def getEpochsNumber: Int = $(this.epochsNumber)

  /** Path to folder to output logs (Default: `""`)
   * If no path is specified, no logs are generated
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

  /** Choose the proportion of training dataset to be validated against the model on each Epoch (Default: `0.0f`).
   * The value should be between `0.0` and `1.0` and by default it is `0.0` and off.
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
    * fat rows (Default: `false`)
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
