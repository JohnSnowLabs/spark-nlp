package com.johnsnowlabs.nlp.annotators.ner.dl

import java.io.File

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, NAMED_ENTITY, TOKEN, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.common.{NerTagged, WordpieceEmbeddingsSentence}
import com.johnsnowlabs.nlp.annotators.ner.{NerApproach, Verbose}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.commons.lang.SystemUtils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import org.tensorflow.Session

import scala.util.Random


class NerDLApproach(override val uid: String)
  extends AnnotatorApproach[NerDLModel]
    with NerApproach[NerDLApproach]
    with Logging {

  def this() = this(Identifiable.randomUID("NerDL"))

  override def getLogName: String = "NerDL"
  override val description = "Trains Tensorflow based Char-CNN-BLSTM model"
  override val inputAnnotatorTypes = Array(DOCUMENT, TOKEN, WORD_EMBEDDINGS)
  override val outputAnnotatorType = NAMED_ENTITY

  val lr = new FloatParam(this, "lr", "Learning Rate")
  val po = new FloatParam(this, "po", "Learning rate decay coefficient. Real Learning Rage = lr / (1 + po * epoch)")
  val batchSize = new IntParam(this, "batchSize", "Batch size")
  val dropout = new FloatParam(this, "dropout", "Dropout coefficient")

  def setLr(lr: Float) = set(this.lr, lr)
  def setPo(po: Float) = set(this.po, po)
  def setBatchSize(batch: Int) = set(this.batchSize, batch)
  def setDropout(dropout: Float) = set(this.dropout, dropout)

  setDefault(
    minEpochs -> 0,
    maxEpochs -> 70,
    lr -> 1e-3f,
    po -> 0.005f,
    batchSize -> 32,
    dropout -> 0.5f,
    verbose -> Verbose.Silent.id
  )

  override val verboseLevel = Verbose($(verbose))


  def calculateEmbeddingsDim(sentences: Seq[WordpieceEmbeddingsSentence]): Int = {
    sentences.find(s => s.tokens.nonEmpty)
      .map(s => s.tokens.head.embeddings.length)
      .getOrElse(1)
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

    //Use CPU
    //val config = Array[Byte](10, 7, 10, 3, 67, 80, 85, 16, 0)
    //Use GPU
    val config = Array[Byte](56, 1)
    val graphFile = NerDLApproach.searchForSuitableGraph(labels.length, embeddingsDim, chars.length)
    val graph = TensorflowWrapper.readGraph(graphFile)

    val session = new Session(graph, config)

    val tf = new TensorflowWrapper(session, graph)

    val ner = try {
      val model = new TensorflowNer(tf, encoder, $(batchSize), Verbose($(verbose)))
      if (isDefined(randomSeed)) {
        Random.setSeed($(randomSeed))
      }

      model.train(trainDataset, $(lr), $(po), $(batchSize), $(dropout), 0, $(maxEpochs))
      model
    }

    catch {
      case e: Exception =>
        session.close()
        graph.close()
        throw e
    }

    new NerDLModel()
      .setTensorflow(tf)
      .setDatasetParams(ner.encoder.params)
      .setBatchSize($(batchSize))
  }
}

trait WithGraphResolver  {
  def searchForSuitableGraph(tags: Int, embeddingsNDims: Int, nChars: Int): String = {
    val files = ResourceHelper.listResourceDirectory("/ner-dl")

    // 1. Filter Graphs by embeddings
    val embeddingsFiltered = files.map { filePath =>
      val file = new File(filePath)
      val name = file.getName

      val graphPrefix = if (SystemUtils.IS_OS_WINDOWS) "blstm-noncontrib_" else "blstm_"

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

    require(embeddingsFiltered.exists(_.nonEmpty), s"Not found tensorflow graph suitable for embeddings dim: $embeddingsNDims. " +
      s"Generate graph by python code before usage.")

    // 2. Filter by labels and nChars
    val tagsFiltered = embeddingsFiltered.map {
      case Some((fileTags, fileEmbeddingsNDims, fileNChars)) =>
        if (tags > fileTags)
          None
        else
          Some((fileTags, fileEmbeddingsNDims, fileNChars))
      case _ => None
    }

    require(tagsFiltered.exists(_.nonEmpty), s"Not found tensorflow graph suitable for number of tags: $tags. " +
      s"Generate graph by python code before usage.")

    // 3. Filter by labels and nChars
    val charsFiltered = tagsFiltered.map {
      case Some((fileTags, fileEmbeddingsNDims, fileNChars)) =>
        if (nChars > fileNChars)
          None
        else
          Some((fileTags, fileEmbeddingsNDims, fileNChars))
      case _ => None
    }

    require(charsFiltered.exists(_.nonEmpty), s"Not found tensorflow graph suitable for number of chars: $nChars. " +
      s"Generate graph by python code before usage.")

    for (i <- files.indices) {
      if (charsFiltered(i).nonEmpty)
        return files(i)
    }

    throw new IllegalStateException("Code shouldn't pass here")
  }
}

object NerDLApproach extends DefaultParamsReadable[NerDLApproach] with WithGraphResolver
