package com.johnsnowlabs.nlp.annotators.ner.dl

import java.io.File

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.{AnnotatorApproach, ParamsAndFeaturesWritable}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, NAMED_ENTITY, TOKEN, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.common.{NerTagged, WordpieceEmbeddingsSentence}
import com.johnsnowlabs.nlp.annotators.ner.{NerApproach, Verbose}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.commons.io.IOUtils
import org.apache.commons.lang.SystemUtils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.tensorflow.Graph

import scala.util.Random

class NerDLApproach(override val uid: String)
  extends AnnotatorApproach[NerDLModel]
    with NerApproach[NerDLApproach]
    with Logging
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("NerDL"))

  override def getLogName: String = "NerDL"
  override val description = "Trains Tensorflow based Char-CNN-BLSTM model"
  override val inputAnnotatorTypes = Array(DOCUMENT, TOKEN, WORD_EMBEDDINGS)
  override val outputAnnotatorType:String = NAMED_ENTITY

  val lr = new FloatParam(this, "lr", "Learning Rate")
  val po = new FloatParam(this, "po", "Learning rate decay coefficient. Real Learning Rage = lr / (1 + po * epoch)")
  val batchSize = new IntParam(this, "batchSize", "Batch size")
  val dropout = new FloatParam(this, "dropout", "Dropout coefficient")
  val graphFolder = new Param[String](this, "graphFolder", "Folder path that contain external graph files")
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")
  val useContrib = new BooleanParam(this, "useContrib", "whether to use contrib LSTM Cells. Not compatible with Windows. Might slightly improve accuracy.")
  val validationSplit = new FloatParam(this, "validationSplit", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.")
  val evaluationLogExtended = new BooleanParam(this, "evaluationLogExtended", "Whether logs for validation to be extended: it displays time and evaluation of each label. Default is false.")
  val enableOutputLogs = new BooleanParam(this, "enableOutputLogs", "Whether to output to annotators log folder")
  val testDataset = new ExternalResourceParam(this, "testDataset", "Path to test dataset. " +
    "If set used to calculate statistic on it during training.")
  val includeConfidence = new BooleanParam(this, "includeConfidence", "whether to include confidence scores in annotation metadata")

  def getLr: Float = $(this.lr)
  def getPo: Float = $(this.po)
  def getBatchSize: Int = $(this.batchSize)
  def getDropout: Float = $(this.dropout)
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))
  def getUseContrib: Boolean = $(this.useContrib)
  def getValidationSplit: Float = $(this.validationSplit)
  def getIncludeConfidence: Boolean = $(includeConfidence)
  def getEnableOutputLogs: Boolean = $(enableOutputLogs)

  def setLr(lr: Float):NerDLApproach.this.type = set(this.lr, lr)
  def setPo(po: Float):NerDLApproach.this.type = set(this.po, po)
  def setBatchSize(batch: Int):NerDLApproach.this.type = set(this.batchSize, batch)
  def setDropout(dropout: Float):NerDLApproach.this.type = set(this.dropout, dropout)
  def setGraphFolder(path: String):NerDLApproach.this.type = set(this.graphFolder, path)
  def setConfigProtoBytes(bytes: Array[Int]):NerDLApproach.this.type = set(this.configProtoBytes, bytes)
  def setUseContrib(value: Boolean):NerDLApproach.this.type = if (value && SystemUtils.IS_OS_WINDOWS) throw new UnsupportedOperationException("Cannot set contrib in Windows") else set(useContrib, value)
  def setValidationSplit(validationSplit: Float):NerDLApproach.this.type = set(this.validationSplit, validationSplit)

  def setEvaluationLogExtended(evaluationLogExtended: Boolean):NerDLApproach.this.type = set(this.evaluationLogExtended, evaluationLogExtended)
  def setEnableOutputLogs(enableOutputLogs: Boolean):NerDLApproach.this.type = set(this.enableOutputLogs, enableOutputLogs)
  def setTestDataset(path: String,
                     readAs: ReadAs.Format = ReadAs.SPARK_DATASET,
                     options: Map[String, String] = Map("format" -> "parquet")): this.type =
    set(testDataset, ExternalResource(path, readAs, options))

  def setTestDataset(er: ExternalResource):NerDLApproach.this.type = set(testDataset, er)
  def setIncludeConfidence(value: Boolean):NerDLApproach.this.type = set(this.includeConfidence, value)

  setDefault(
    minEpochs -> 0,
    maxEpochs -> 70,
    lr -> 1e-3f,
    po -> 0.005f,
    batchSize -> 8,
    dropout -> 0.5f,
    verbose -> Verbose.Silent.id,
    useContrib -> {if (SystemUtils.IS_OS_WINDOWS) false else true},
    validationSplit -> 0.0f,
    evaluationLogExtended -> false,
    includeConfidence -> false,
    enableOutputLogs -> false
  )

  override val verboseLevel = Verbose($(verbose))

  def calculateEmbeddingsDim(sentences: Seq[WordpieceEmbeddingsSentence]): Int = {
    sentences.find(s => s.tokens.nonEmpty)
      .map(s => s.tokens.head.embeddings.length)
      .getOrElse(1)
  }

  override def beforeTraining(spark: SparkSession): Unit = {
    LoadsContrib.loadContribToCluster(spark)
    LoadsContrib.loadContribToTensorflow()
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NerDLModel = {

    require($(validationSplit) <= 1f | $(validationSplit) >= 0f, "The validationSplit must be between 0f and 1f")

    val train = dataset.toDF()

    val test = if (!isDefined(testDataset)) {
      val emptyValid: Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)] = Array.empty
      emptyValid
    }
    else {
      val testDataFrame = ResourceHelper.readParquetSparkDatFrame($(testDataset))
      NerTagged.collectTrainingInstances(testDataFrame, getInputCols, $(labelColumn))
    }

    val trainDataset = NerTagged.collectTrainingInstances(train, getInputCols, $(labelColumn))
    val trainSentences = trainDataset.map(r => r._2)

    val labels = trainDataset.flatMap(r => r._1.labels).distinct
    val chars = trainDataset.flatMap(r => r._2.tokens.flatMap(token => token.token.toCharArray)).distinct
    val embeddingsDim = calculateEmbeddingsDim(trainSentences)

    val settings = DatasetEncoderParams(labels.toList, chars.toList,
      Array.fill(embeddingsDim)(0f).toList, embeddingsDim)
    val encoder = new NerDatasetEncoder(
      settings
    )

    val graphFile = NerDLApproach.searchForSuitableGraph(labels.length, embeddingsDim, chars.length + 1, get(graphFolder), getUseContrib)

    val graph = new Graph()
    val graphStream = ResourceHelper.getResourceStream(graphFile)
    val graphBytesDef = IOUtils.toByteArray(graphStream)
    graph.importGraphDef(graphBytesDef)

    val tf = new TensorflowWrapper(Variables(Array.empty[Byte], Array.empty[Byte]), graph.toGraphDef)

    val ner = try {
      val model = new TensorflowNer(tf, encoder, $(batchSize), Verbose($(verbose)))
      if (isDefined(randomSeed)) {
        Random.setSeed($(randomSeed))
      }

      model.train(trainDataset,
        $(lr),
        $(po),
        $(batchSize),
        $(dropout),
        graphFileName = graphFile,
        test = test,
        endEpoch = $(maxEpochs),
        configProtoBytes=getConfigProtoBytes,
        validationSplit=$(validationSplit),
        evaluationLogExtended=$(evaluationLogExtended),
        includeConfidence=$(includeConfidence),
        enableOutputLogs=$(enableOutputLogs),
        uuid=this.uid
      )
      model
    }

    catch {
      case e: Exception =>
        graph.close()
        throw e
    }

    val newWrapper = new TensorflowWrapper(TensorflowWrapper.extractVariables(tf.getSession(configProtoBytes=getConfigProtoBytes)), tf.graph)

    val model = new NerDLModel()
      .setDatasetParams(ner.encoder.params)
      .setBatchSize($(batchSize))
      .setModelIfNotSet(dataset.sparkSession, newWrapper)
      .setIncludeConfidence($(includeConfidence))

    if (get(configProtoBytes).isDefined)
      model.setConfigProtoBytes($(configProtoBytes))

    model

  }
}

trait WithGraphResolver  {
  def searchForSuitableGraph(tags: Int, embeddingsNDims: Int, nChars: Int, localGraphPath: Option[String] = None, loadContrib: Boolean = false): String = {
    val files = localGraphPath.map(path => ResourceHelper.listLocalFiles(ResourceHelper.copyToLocal(path)).map(_.getAbsolutePath))
      .getOrElse(ResourceHelper.listResourceDirectory("/ner-dl"))

    // 1. Filter Graphs by embeddings
    val embeddingsFiltered = files.map { filePath =>
      val file = new File(filePath)
      val name = file.getName

      val graphPrefix = if (loadContrib) "blstm_" else "blstm-noncontrib_"

      if (name.startsWith(graphPrefix)) {
        val clean = name.replace(graphPrefix, "").replace(".pb", "")
        val graphParams = clean.split("_").take(4).map(s => s.toInt)
        val Array(fileTags, fileEmbeddingsNDims, _, fileNChars) = graphParams

        if (embeddingsNDims == fileEmbeddingsNDims)
          Some((fileTags, fileEmbeddingsNDims, fileNChars))
        else
          None
      }
      else {
        None
      }
    }

    require(embeddingsFiltered.exists(_.nonEmpty), s"Graph dimensions should be $embeddingsNDims: Could not find a suitable tensorflow graph for embeddings dim: $embeddingsNDims tags: $tags nChars: $nChars. " +
      s"Check https://nlp.johnsnowlabs.com/docs/en/graph for instructions to generate the required graph.")

    // 2. Filter by labels and nChars
    val tagsFiltered = embeddingsFiltered.map {
      case Some((fileTags, fileEmbeddingsNDims, fileNChars)) =>
        if (tags > fileTags)
          None
        else
          Some((fileTags, fileEmbeddingsNDims, fileNChars))
      case _ => None
    }

    require(tagsFiltered.exists(_.nonEmpty), s"Graph tags size should be $tags: Could not find a suitable tensorflow graph for embeddings dim: $embeddingsNDims tags: $tags nChars: $nChars. " +
      s"Check https://nlp.johnsnowlabs.com/docs/en/graph for instructions to generate the required graph.")

    // 3. Filter by labels and nChars
    val charsFiltered = tagsFiltered.map {
      case Some((fileTags, fileEmbeddingsNDims, fileNChars)) =>
        if (nChars > fileNChars)
          None
        else
          Some((fileTags, fileEmbeddingsNDims, fileNChars))
      case _ => None
    }

    require(charsFiltered.exists(_.nonEmpty), s"Graph chars size should be $nChars: Could not find a suitable tensorflow graph for embeddings dim: $embeddingsNDims tags: $tags nChars: $nChars. " +
      s"Check https://nlp.johnsnowlabs.com/docs/en/graph for instructions to generate the required graph")

    for (i <- files.indices) {
      if (charsFiltered(i).nonEmpty)
        return files(i)
    }

    throw new IllegalStateException("Code shouldn't pass here")
  }
}

object NerDLApproach extends DefaultParamsReadable[NerDLApproach] with WithGraphResolver