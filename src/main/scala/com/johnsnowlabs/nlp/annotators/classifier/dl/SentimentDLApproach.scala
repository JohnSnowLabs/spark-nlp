package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType, ParamsAndFeaturesWritable}
import com.johnsnowlabs.nlp.AnnotatorType.{CATEGORY, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntArrayParam, IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, StringType}
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.util.Random

class SentimentDLApproach(override val uid: String)
  extends AnnotatorApproach[SentimentDLModel]
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("SentimentDL"))

  override val description = "Trains TensorFlow model for Sentiment Classification"
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(SENTENCE_EMBEDDINGS)
  override val outputAnnotatorType: String = CATEGORY

  val randomSeed = new IntParam(this, "randomSeed", "Random seed")

  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each document")
  val lr = new FloatParam(this, "lr", "Learning Rate")
  val batchSize = new IntParam(this, "batchSize", "Batch size")
  val dropout = new FloatParam(this, "dropout", "Dropout coefficient")
  val threshold = new FloatParam(this, "threshold", "The minimum threshold for the final result otheriwse it will be either neutral or the value set in thresholdLabel.")
  val thresholdLabel = new Param[String](this, "thresholdLabel", "In case the score is less than threshold, what should be the label. Default is neutral.")
  val maxEpochs = new IntParam(this, "maxEpochs", "Maximum number of epochs to train")
  val enableOutputLogs = new BooleanParam(this, "enableOutputLogs", "Whether to output to annotators log folder")
  val outputLogsPath = new Param[String](this, "outputLogsPath", "Folder path to save training logs")
  val validationSplit = new FloatParam(this, "validationSplit", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.")
  val verbose = new IntParam(this, "verbose", "Level of verbosity during training")
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()"
  )

  def setLabelColumn(column: String): SentimentDLApproach.this.type = set(labelColumn, column)
  def setLr(lr: Float): SentimentDLApproach.this.type = set(this.lr, lr)
  def setBatchSize(batch: Int): SentimentDLApproach.this.type = set(this.batchSize, batch)
  def setDropout(dropout: Float): SentimentDLApproach.this.type = set(this.dropout, dropout)
  def setThreshold(threshold: Float): SentimentDLApproach.this.type = set(this.threshold, threshold)
  def setThresholdLabel(label: String):SentimentDLApproach.this.type = set(this.thresholdLabel, label)
  def setMaxEpochs(epochs: Int): SentimentDLApproach.this.type = set(maxEpochs, epochs)
  def setConfigProtoBytes(bytes: Array[Int]): SentimentDLApproach.this.type = set(this.configProtoBytes, bytes)
  def setEnableOutputLogs(enableOutputLogs: Boolean): SentimentDLApproach.this.type = set(this.enableOutputLogs, enableOutputLogs)
  def setOutputLogsPath(path: String):SentimentDLApproach.this.type = set(this.outputLogsPath, path)
  def setValidationSplit(validationSplit: Float):SentimentDLApproach.this.type = set(this.validationSplit, validationSplit)
  def setVerbose(verbose: Int): SentimentDLApproach.this.type = set(this.verbose, verbose)
  def setVerbose(verbose: Verbose.Level): SentimentDLApproach.this.type = set(this.verbose, verbose.id)

  def getLabelColumn: String = $(this.labelColumn)
  def getLr: Float = $(this.lr)
  def getBatchSize: Int = $(this.batchSize)
  def getDropout: Float = $(this.dropout)
  def getThreshold: Float = $(this.threshold)
  def getThresholdLabel: String = $(this.thresholdLabel)
  def getEnableOutputLogs: Boolean = $(enableOutputLogs)
  def getOutputLogsPath: String = $(outputLogsPath)
  def getValidationSplit: Float = $(this.validationSplit)
  def getMaxEpochs: Int = $(maxEpochs)
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  setDefault(
    maxEpochs -> 10,
    lr -> 5e-3f,
    dropout -> 0.5f,
    batchSize -> 64,
    enableOutputLogs -> false,
    verbose -> Verbose.Silent.id,
    validationSplit -> 0.0f,
    outputLogsPath -> "",
    threshold -> 0.6f,
    thresholdLabel -> "neutral"
  )

  override def beforeTraining(spark: SparkSession): Unit = {}

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): SentimentDLModel = {

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

    require(
      labels.length == 2,
      s"The total unique number of classes must be 2. Currently is ${labels.length}. Please use ClassifierDLApproach" +
        s" if you have more than 2 classes/labels."
    )

    val tf = loadSavedModel()

    val settings = ClassifierDatasetEncoderParams(
      tags = labels
    )

    val encoder = new ClassifierDatasetEncoder(
      settings
    )

    val embeddingsDim = encoder.calculateEmbeddingsDim(train)
    require(
      embeddingsDim <= 1024,
      s"The SentimentDL only accepts embeddings less than 1024 dimensions. Current dimension is ${embeddingsDim}. Please use embeddings" +
        s" with less than "
    )
    val trainDataset = encoder.collectTrainingInstances(train, getLabelColumn)
    val inputEmbeddings = encoder.extractSentenceEmbeddings(trainDataset)
    val inputLabels = encoder.extractLabels(trainDataset)

    val classifier = try {
      val model = new TensorflowSentiment(
        tensorflow = tf,
        encoder,
        Verbose($(verbose))
      )
      if (isDefined(randomSeed)) {
        Random.setSeed($(randomSeed))
      }

      model.train(
        inputEmbeddings,
        inputLabels,
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

    val newWrapper = new TensorflowWrapper(TensorflowWrapper.extractVariablesSavedModel(tf.getSession(configProtoBytes = getConfigProtoBytes)), tf.graph)

    val model = new SentimentDLModel()
      .setDatasetParams(classifier.encoder.params)
      .setModelIfNotSet(dataset.sparkSession, newWrapper)
      .setStorageRef(embeddingsRef)
      .setThreshold($(threshold))
      .setThresholdLabel($(thresholdLabel))

    if (get(configProtoBytes).isDefined)
      model.setConfigProtoBytes($(configProtoBytes))

    model
  }

  def loadSavedModel(): TensorflowWrapper = {

    val wrapper =
      TensorflowWrapper.readZippedSavedModel("/sentiment-dl", tags = Array("serve"), initAllTables = true)
    wrapper
  }
}

object SentimentApproach extends DefaultParamsReadable[SentimentDLApproach]
