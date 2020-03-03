package com.johnsnowlabs.nlp.annotators.classifier.dl

import java.io.File

import com.johnsnowlabs.ml.tensorflow.{ClassifierDatasetEncoder, ClassifierDatasetEncoderParams, TensorflowClassifier, TensorflowWrapper}
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType, ParamsAndFeaturesWritable}
import com.johnsnowlabs.nlp.AnnotatorType.{CATEGORY, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntArrayParam, IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.types.{IntegerType, StringType}
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.util.Random

class ClassifierDLApproach(override val uid: String)
  extends AnnotatorApproach[ClassifierDLModel]
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("ClassifierDL"))

  override val description = "Trains TensorFlow model for multi-class text classification"
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(SENTENCE_EMBEDDINGS)
  override val outputAnnotatorType: String = CATEGORY

  val randomSeed = new IntParam(this, "randomSeed", "Random seed")

  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each document")
  val lr = new FloatParam(this, "lr", "Learning Rate")
  val batchSize = new IntParam(this, "batchSize", "Batch size")
  val dropout = new FloatParam(this, "dropout", "Dropout coefficient")
  val maxEpochs = new IntParam(this, "maxEpochs", "Maximum number of epochs to train")
  val enableOutputLogs = new BooleanParam(this, "enableOutputLogs", "Whether to output to annotators log folder")
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()"
  )

  def setLabelColumn(column: String): ClassifierDLApproach.this.type = set(labelColumn, column)
  def setLr(lr: Float): ClassifierDLApproach.this.type = set(this.lr, lr)
  def setBatchSize(batch: Int): ClassifierDLApproach.this.type = set(this.batchSize, batch)
  def setDropout(dropout: Float): ClassifierDLApproach.this.type = set(this.dropout, dropout)
  def setMaxEpochs(epochs: Int): ClassifierDLApproach.this.type = set(maxEpochs, epochs)
  def setConfigProtoBytes(bytes: Array[Int]): ClassifierDLApproach.this.type = set(this.configProtoBytes, bytes)

  def getLabelColumn: String = $(this.labelColumn)
  def getLr: Float = $(this.lr)
  def getBatchSize: Int = $(this.batchSize)
  def getDropout: Float = $(this.dropout)
  def getEnableOutputLogs: Boolean = $(enableOutputLogs)
  def getMaxEpochs: Int = $(maxEpochs)
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  setDefault(
    maxEpochs -> 30,
    lr -> 5e-3f,
    dropout -> 0.5f,
    batchSize -> 64,
    enableOutputLogs -> false
  )

  override def beforeTraining(spark: SparkSession): Unit = {}

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): ClassifierDLModel = {

    val labelColType = dataset.schema($(labelColumn)).dataType
    require(
      labelColType != StringType | labelColType != IntegerType,
      s"The label column $labelColumn must be either a single StringType or IntegerType"
    )

    val embeddingsRef = HasStorageRef.getStorageRefFromInput(dataset, $(inputCols), AnnotatorType.SENTENCE_EMBEDDINGS)

    val embeddingsField: String = ".embeddings"
    val inputColumns = (getInputCols(0) + embeddingsField)
    val train = dataset.select(dataset.col($(labelColumn)).cast("string"), dataset.col(inputColumns))
    val labels = train.select($(labelColumn)).distinct.collect.map(x => x(0).toString)

    val tf = loadSavedModel("src/main/resources/classifier-dl/multi-class")

    val settings = ClassifierDatasetEncoderParams(
      tags = labels
    )

    val encoder = new ClassifierDatasetEncoder(
      settings
    )


    val trainDataset = encoder.collectTrainingInstances(train, getLabelColumn)
    val inputEmbeddings = encoder.extractSentenceEmbeddings(trainDataset)
    val inputLabels = encoder.extractLabels(trainDataset)

    val classifier = try {
      val model = new TensorflowClassifier(
        tensorflow = tf,
        encoder
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
        validationSplit = 0.1f,
        evaluationLogExtended = true,
        uuid = this.uid
      )
      model
    } catch {
      case e: Exception =>
        throw e
    }

    val model = new ClassifierDLModel()
      .setDatasetParams(classifier.encoder.params)
      .setModelIfNotSet(dataset.sparkSession, tf)
      .setStorageRef(embeddingsRef)

    if (get(configProtoBytes).isDefined)
      model.setConfigProtoBytes($(configProtoBytes))

    model
  }

  def loadSavedModel(folder: String): TensorflowWrapper = {

    val f = new File(folder)
    val savedModel = new File(folder, "saved_model.pb")
    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $folder"
    )

    val wrapper =
      TensorflowWrapper.read(folder, zipped = false, useBundle = true, tags = Array("serve"), initAllTables = true)
    wrapper
  }
}

object NerDLApproach extends DefaultParamsReadable[ClassifierDLApproach]
