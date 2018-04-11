package com.johnsnowlabs.nlp.annotators.ner.dl

import java.io.File

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, NAMED_ENTITY, TOKEN}
import com.johnsnowlabs.nlp.{AnnotatorType, DocumentAssembler, HasRecursiveFit}
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.{NerTagged, TokenizedSentence}
import com.johnsnowlabs.nlp.annotators.ner.{NerApproach, Verbose}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.datasets.CoNLL
import com.johnsnowlabs.nlp.embeddings.ApproachWithWordEmbeddings
import com.johnsnowlabs.nlp.util.io.ResourceHelper.SourceStream
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.commons.io.IOUtils
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.tensorflow.{Graph, Session}

import scala.util.Random


class NerDLApproach(override val uid: String)
  extends ApproachWithWordEmbeddings[NerDLApproach, NerDLModel]
    with HasRecursiveFit[NerDLModel]
    with NerApproach[NerDLApproach]
    with Logging {

  def this() = this(Identifiable.randomUID("NerDL"))

  override def getLogName: String = "NerDL"
  override val description = "Trains Tensorflow based Char-CNN-BLSTM model"
  override val requiredAnnotatorTypes = Array(DOCUMENT, TOKEN)
  override val annotatorType = NAMED_ENTITY

  val lr = new FloatParam(this, "lr", "Learning Rate")
  val po = new FloatParam(this, "po", "Learning rate decay coefficient. Real Learning Rage = lr / (1 + po * epoch)")
  val batchSize = new IntParam(this, "batchSize", "Batch size")
  val dropout = new FloatParam(this, "dropout", "Dropout coefficient")

  val validationDataset = new ExternalResourceParam(this, "validationDataset", "Path to validation dataset. " +
    "If set used to calculate statistic on it during training.")
  val testDataset = new ExternalResourceParam(this, "testDataset", "Path to test dataset. " +
    "If set used to calculate statistic on it during training.")

  def setLr(lr: Float) = set(this.lr, lr)
  def setPo(po: Float) = set(this.po, po)
  def setBatchSize(batch: Int) = set(this.batchSize, batch)
  def setDropout(dropout: Float) = set(this.dropout, dropout)

  def setValidationDataset(path: String,
                         readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                         options: Map[String, String] = Map("format" -> "text")): this.type =
    set(validationDataset, ExternalResource(path, readAs, options))

  def setValidationDataset(er: ExternalResource) = set(validationDataset, er)

  def setTestDataset(path: String,
                            readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                            options: Map[String, String] = Map("format" -> "text")): this.type =
    set(testDataset, ExternalResource(path, readAs, options))

  def setTestDataset(er: ExternalResource) = set(testDataset, er)

  setDefault(
    minEpochs -> 0,
    maxEpochs -> 50,
    lr -> 0.2f,
    po -> 0.05f,
    batchSize -> 9,
    dropout -> 0.5f,
    verbose -> Verbose.Silent.id
  )

  override val verboseLevel = Verbose($(verbose))

  private def getTrainDataframe(dataset: Dataset[_], recursivePipeline: Option[PipelineModel])
    :(DataFrame, Option[DataFrame], Option[DataFrame]) = {

    lazy val pipelineModel = recursivePipeline.getOrElse {

      logger.warn("NER DL not in a RecursivePipeline. " +
        "It is recommended to use a com.jonsnowlabs.nlp.RecursivePipeline for " +
        "better performance during training")

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val sentenceDetector = new SentenceDetector()
        .setCustomBoundChars(Array("\n\n", "\n\r\n\r"))
        .setInputCols(Array("document"))
        .setOutputCol("sentence")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val pipeline = new Pipeline().setStages(
        Array(
          documentAssembler,
          sentenceDetector,
          tokenizer)
      )

      pipeline.fit(dataset)
    }

    val reader = CoNLL(3, AnnotatorType.NAMED_ENTITY)

    val train = if (!isDefined(externalDataset))
      dataset.toDF()
    else
      pipelineModel.transform(reader.readDataset($(externalDataset), dataset.sparkSession).toDF)

    val valid = if (!isDefined(validationDataset))
      None
    else
      Some(pipelineModel.transform(reader.readDataset($(validationDataset), dataset.sparkSession).toDF))

    val test = if (!isDefined(testDataset))
      None
    else
      Some(pipelineModel.transform(reader.readDataset($(testDataset), dataset.sparkSession).toDF))

    (train, valid, test)
  }


  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NerDLModel = {
    require(isDefined(sourceEmbeddingsPath), "embeddings must be set before training")

    val (train, valid, test) = getTrainDataframe(dataset, recursivePipeline)

    val trainDataset = NerTagged.collectTrainingInstances(train, getInputCols, $(labelColumn))

    val validationDataset =
      if (valid.isEmpty) Array.empty[(TextSentenceLabels, TokenizedSentence)]
    else
      NerTagged.collectTrainingInstances(valid.get, getInputCols, $(labelColumn))

    val testDataset =
      if (test.isEmpty) Array.empty[(TextSentenceLabels, TokenizedSentence)]
      else
        NerTagged.collectTrainingInstances(test.get, getInputCols, $(labelColumn))


    val labels = trainDataset.flatMap(r => r._1.labels).distinct
    val chars = trainDataset.flatMap(r => r._2.tokens.flatMap(token => token.toCharArray)).distinct

    val settings = DatasetEncoderParams(labels.toList, chars.toList)
    val encoder = new NerDatasetEncoder(
      embeddings.get.getEmbeddings,
      settings
    )

    val graph = new Graph()
    val session = new Session(graph)

    val graphFile = NerDLApproach.searchForSuitableGraph(labels.length, $(embeddingsNDims), chars.length)

    val graphStream = ResourceHelper.getResourceStream(graphFile)
    val graphBytesDef = IOUtils.toByteArray(graphStream)
    graph.importGraphDef(graphBytesDef)

    val tf = new TensorflowWrapper(session, graph)

    val ner = try {
      val model = new TensorflowNer(tf, encoder, $(batchSize), Verbose($(verbose)))
      if (isDefined(randomSeed)) {
        Random.setSeed($(randomSeed))
      }

      model.train(trainDataset, $(lr), $(po), $(batchSize), $(dropout), 0, $(maxEpochs), validationDataset, testDataset)
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

      if (name.startsWith("char_cnn_blstm_")) {
        val clean = name.replace("char_cnn_blstm_", "").replace(".pb", "")
        val graphParams = clean.split("_").take(3).map(s => s.toInt)
        val Array(fileTags, fileEmbeddingsNDims, fileNChars) = graphParams

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

    for (i <- 0 until files.length) {
      if (charsFiltered(i).nonEmpty)
        return files(i)
    }

    throw new IllegalStateException("Code shouldn't pass here")
  }
}

object NerDLApproach extends DefaultParamsReadable[NerDLApproach] with WithGraphResolver
