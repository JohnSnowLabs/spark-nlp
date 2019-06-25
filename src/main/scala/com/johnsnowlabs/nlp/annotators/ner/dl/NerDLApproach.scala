package com.johnsnowlabs.nlp.annotators.ner.dl

import java.io.File

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.{AnnotatorApproach, ParamsAndFeaturesWritable}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, NAMED_ENTITY, TOKEN, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.common.{NerTagged, WordpieceEmbeddingsSentence}
import com.johnsnowlabs.nlp.annotators.ner.{NerApproach, Verbose}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
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
  val trainValidationProp = new FloatParam(this, "trainValidationProp", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.")

  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  def getUseContrib(): Boolean = $(this.useContrib)

  def setLr(lr: Float):NerDLApproach.this.type = set(this.lr, lr)
  def setPo(po: Float):NerDLApproach.this.type = set(this.po, po)
  def setBatchSize(batch: Int):NerDLApproach.this.type = set(this.batchSize, batch)
  def setDropout(dropout: Float):NerDLApproach.this.type = set(this.dropout, dropout)
  def setGraphFolder(path: String):NerDLApproach.this.type = set(this.graphFolder, path)
  def setConfigProtoBytes(bytes: Array[Int]):NerDLApproach.this.type = set(this.configProtoBytes, bytes)
  def setUseContrib(value: Boolean):NerDLApproach.this.type = if (value && SystemUtils.IS_OS_WINDOWS) throw new UnsupportedOperationException("Cannot set contrib in Windows") else set(useContrib, value)
  def setTrainValidationProp(trainValidationProp: Float):NerDLApproach.this.type = set(this.trainValidationProp, trainValidationProp)

  setDefault(
    minEpochs -> 0,
    maxEpochs -> 70,
    lr -> 1e-3f,
    po -> 0.005f,
    batchSize -> 8,
    dropout -> 0.5f,
    verbose -> Verbose.Silent.id,
    useContrib -> {if (SystemUtils.IS_OS_WINDOWS) false else true},
    trainValidationProp -> 0.0f
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

    val train = dataset.toDF()

    val trainDataset = NerTagged.collectTrainingInstances(train, getInputCols, $(labelColumn))
    val trainSentences = trainDataset.map(r => r._2)

    val labels = trainDataset.flatMap(r => r._1.labels).distinct ++ Set("X")
    val chars = trainDataset.flatMap(r => r._2.tokens.flatMap(token => token.wordpiece.toCharArray)).distinct
    val embeddingsDim = calculateEmbeddingsDim(trainSentences)

    val settings = DatasetEncoderParams(labels.toList, chars.toList,
      Array.fill(embeddingsDim)(0f).toList, embeddingsDim)
    val encoder = new NerDatasetEncoder(
      settings
    )

    val graphFile = NerDLApproach.searchForSuitableGraph(labels.length, embeddingsDim, chars.length, get(graphFolder), getUseContrib())

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

      model.train(trainDataset, $(lr), $(po), $(batchSize), $(dropout), 0, $(maxEpochs), configProtoBytes=getConfigProtoBytes, trainValidationProp=$(trainValidationProp))
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

    require(embeddingsFiltered.exists(_.nonEmpty), s"Could not find a suitable tensorflow graph for embeddings dim: $embeddingsNDims tags: $tags nChars: $nChars. " +
      s"Generate graph by python code in python/tensorflow/ner/create_models  before usage and use setGraphFolder Param to point to output.")

    // 2. Filter by labels and nChars
    val tagsFiltered = embeddingsFiltered.map {
      case Some((fileTags, fileEmbeddingsNDims, fileNChars)) =>
        if (tags > fileTags)
          None
        else
          Some((fileTags, fileEmbeddingsNDims, fileNChars))
      case _ => None
    }

    require(tagsFiltered.exists(_.nonEmpty), s"Not found tensorflow graph suitable for number of dim: $embeddingsNDims tags: $tags nChars: $nChars. " +
      s"Generate graph by python code in python/tensorflow/ner/create_models before usage and use setGraphFolder Param to point to output.")

    // 3. Filter by labels and nChars
    val charsFiltered = tagsFiltered.map {
      case Some((fileTags, fileEmbeddingsNDims, fileNChars)) =>
        if (nChars > fileNChars)
          None
        else
          Some((fileTags, fileEmbeddingsNDims, fileNChars))
      case _ => None
    }

    require(charsFiltered.exists(_.nonEmpty), s"Not found tensorflow graph suitable for number of dim: $embeddingsNDims tags: $tags nChars: $nChars. " +
      s"Generate graph by python code before usage.")

    for (i <- files.indices) {
      if (charsFiltered(i).nonEmpty)
        return files(i)
    }

    throw new IllegalStateException("Code shouldn't pass here")
  }
}

object NerDLApproach extends DefaultParamsReadable[NerDLApproach] with WithGraphResolver