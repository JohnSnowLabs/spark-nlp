package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorType.{CATEGORY, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType, ParamsAndFeaturesWritable}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, StringType}
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.util.Random

/**
  * ClassifierDL is a generic Multi-class Text Classification. ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an input for text classifications.
  * The ClassifierDL annotator uses a deep learning model (DNNs) we have built inside TensorFlow and supports up to 100 classes
  *
  * NOTE: This annotator accepts a label column of a single item in either type of String, Int, Float, or Double.
  *
  * NOTE: UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings can be used for the inputCol
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLTestSpec.scala]] for further reference on how to use this API
  *
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
  **/
class ClassifierDLApproach(override val uid: String)
  extends AnnotatorApproach[ClassifierDLModel]
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("ClassifierDL"))

  /** Trains TensorFlow model for multi-class text classification */
  override val description = "Trains TensorFlow model for multi-class text classification"
  /** Input annotator type : SENTENCE_EMBEDDINGS
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(SENTENCE_EMBEDDINGS)
  /** Output annotator type : CATEGORY
    *
    * @group anno
    **/
  override val outputAnnotatorType: String = CATEGORY

  /** Random seed
    *
    * @group param
    **/
  val randomSeed = new IntParam(this, "randomSeed", "Random seed")
  /** Column with label per each document
    *
    * @group param
    **/
  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each document")
  /** Learning Rate
    *
    * @group param
    **/
  val lr = new FloatParam(this, "lr", "Learning Rate")
  /** Batch size
    *
    * @group param
    **/
  val batchSize = new IntParam(this, "batchSize", "Batch size")
  /** Dropout coefficient
    *
    * @group param
    **/
  val dropout = new FloatParam(this, "dropout", "Dropout coefficient")
  /** Maximum number of epochs to train
    *
    * @group param
    **/
  val maxEpochs = new IntParam(this, "maxEpochs", "Maximum number of epochs to train")
  /** Whether to output to annotators log folder
    *
    * @group param
    **/
  val enableOutputLogs = new BooleanParam(this, "enableOutputLogs", "Whether to output to annotators log folder")

  val outputLogsPath = new Param[String](this, "outputLogsPath", "Folder path to save training logs")

  /** Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
    *
    * @group param
    **/
  val validationSplit = new FloatParam(this, "validationSplit", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.")
  /** Level of verbosity during training
    *
    * @group param
    **/
  val verbose = new IntParam(this, "verbose", "Level of verbosity during training")
  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group param
    **/
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** Column with label per each document
    *
    * @group setParam
    **/
  def setLabelColumn(column: String): ClassifierDLApproach.this.type = set(labelColumn, column)

  /** Learning Rate
    *
    * @group setParam
    **/
  def setLr(lr: Float): ClassifierDLApproach.this.type = set(this.lr, lr)

  /** Batch size
    *
    * @group setParam
    **/
  def setBatchSize(batch: Int): ClassifierDLApproach.this.type = set(this.batchSize, batch)

  /** Dropout coefficient
    *
    * @group setParam
    **/
  def setDropout(dropout: Float): ClassifierDLApproach.this.type = set(this.dropout, dropout)

  /** Maximum number of epochs to train
    *
    * @group setParam
    **/
  def setMaxEpochs(epochs: Int): ClassifierDLApproach.this.type = set(maxEpochs, epochs)

  /** Tensorflow config Protobytes passed to the TF session
    *
    * @group setParam
    **/
  def setConfigProtoBytes(bytes: Array[Int]): ClassifierDLApproach.this.type = set(this.configProtoBytes, bytes)

  /** Whether to output to annotators log folder
    *
    * @group setParam
    **/
  def setEnableOutputLogs(enableOutputLogs: Boolean): ClassifierDLApproach.this.type = set(this.enableOutputLogs, enableOutputLogs)


  /** Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
    *
    * @group setParam
    **/
  def setValidationSplit(validationSplit: Float): ClassifierDLApproach.this.type = set(this.validationSplit, validationSplit)

  /** Level of verbosity during training
    *
    * @group setParam
    **/
  def setVerbose(verbose: Int): ClassifierDLApproach.this.type = set(this.verbose, verbose)

  def setOutputLogsPath(path: String):ClassifierDLApproach.this.type = set(this.outputLogsPath, path)

  /** Level of verbosity during training
    *
    * @group setParam
    **/
  def setVerbose(verbose: Verbose.Level): ClassifierDLApproach.this.type = set(this.verbose, verbose.id)

  /** Column with label per each document
    *
    * @group getParam
    **/
  def getLabelColumn: String = $(this.labelColumn)

  /** Learning Rate
    *
    * @group getParam
    **/
  def getLr: Float = $(this.lr)

  /** Batch size
    *
    * @group getParam
    **/
  def getBatchSize: Int = $(this.batchSize)

  /** Dropout coefficient
    *
    * @group getParam
    **/
  def getDropout: Float = $(this.dropout)

  /** Whether to output to annotators log folder
    *
    * @group getParam
    **/
  def getEnableOutputLogs: Boolean = $(enableOutputLogs)

  /** Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
    *
    * @group getParam
    **/
  def getValidationSplit: Float = $(this.validationSplit)

  def getOutputLogsPath: String = $(outputLogsPath)

  /** Maximum number of epochs to train
    *
    * @group getParam
    **/
  def getMaxEpochs: Int = $(maxEpochs)

  /** Tensorflow config Protobytes passed to the TF session
    *
    * @group getParam
    **/
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  setDefault(
    maxEpochs -> 10,
    lr -> 5e-3f,
    dropout -> 0.5f,
    batchSize -> 64,
    enableOutputLogs -> false,
    verbose -> Verbose.Silent.id,
    validationSplit -> 0.0f,
    outputLogsPath -> ""
  )

  override def beforeTraining(spark: SparkSession): Unit = {}

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): ClassifierDLModel = {

    val labelColType = dataset.schema($(labelColumn)).dataType
    require(
      labelColType == StringType | labelColType == IntegerType | labelColType == DoubleType | labelColType == FloatType,
      s"The label column $labelColumn type is $labelColType and it's not compatible. " +
        s"Compatible types are StringType, IntegerType, DoubleType, or FloatType. "
    )

    val embeddingsRef = HasStorageRef.getStorageRefFromInput(dataset, $(inputCols), AnnotatorType.SENTENCE_EMBEDDINGS)

    val embeddingsField: String = ".embeddings"
    val inputColumns = getInputCols(0) + embeddingsField
    val train = dataset.select(dataset.col($(labelColumn)).cast("string"), dataset.col(inputColumns))
    val labels = train.select($(labelColumn)).distinct.collect.map(x => x(0).toString)
    val labelsCount = labels.length

    require(
      labels.length >= 2 && labels.length <= 100,
      s"The total unique number of classes must be more than 2 and less than 100. Currently is ${labels.length}"
    )

    val tfWrapper: TensorflowWrapper = loadSavedModel()

    val settings = ClassifierDatasetEncoderParams(tags = labels)

    val encoder = new ClassifierDatasetEncoder(settings)

    val embeddingsDim = encoder.calculateEmbeddingsDim(train)
    require(
      embeddingsDim <= 1024,
      s"The ClassifierDL only accepts embeddings less than 1024 dimensions. Current dimension is ${embeddingsDim}. Please use embeddings" +
        s" with less than "
    )

    val trainDataset = encoder.collectTrainingInstances(train, getLabelColumn)
    val inputEmbeddings = encoder.extractSentenceEmbeddings(trainDataset)
    val inputLabels = encoder.extractLabels(trainDataset)

    val classifier = try {
      val model = new TensorflowClassifier(
        tensorflow = tfWrapper,
        encoder,
        Verbose($(verbose))
      )
      if (isDefined(randomSeed)) {
        Random.setSeed($(randomSeed))
      }

      model.train(
        inputEmbeddings,
        inputLabels,
        labelsCount,
        lr = $(lr),
        batchSize = $(batchSize),
        dropout = $(dropout),
        endEpoch = $(maxEpochs),
        configProtoBytes = getConfigProtoBytes,
        validationSplit = $(validationSplit),
        enableOutputLogs=$(enableOutputLogs),
        outputLogsPath=$(outputLogsPath),
        uuid = this.uid
      )
      model
    } catch {
      case e: Exception =>
        throw e
    }

    val newWrapper = new TensorflowWrapper(TensorflowWrapper.extractVariablesSavedModel(tfWrapper.getSession(configProtoBytes = getConfigProtoBytes)), tfWrapper.graph)

    val model = new ClassifierDLModel()
      .setDatasetParams(classifier.encoder.params)
      .setModelIfNotSet(dataset.sparkSession, newWrapper)
      .setStorageRef(embeddingsRef)

    if (get(configProtoBytes).isDefined)
      model.setConfigProtoBytes($(configProtoBytes))

    model
  }

  def loadSavedModel(): TensorflowWrapper = {

    val wrapper =
      TensorflowWrapper
        .readZippedSavedModel("/classifier-dl", tags = Array("serve"), initAllTables = true)

    wrapper.variables = Variables(Array.empty[Byte], Array.empty[Byte])
    wrapper
  }
}

object ClassifierDLApproach extends DefaultParamsReadable[ClassifierDLApproach]
