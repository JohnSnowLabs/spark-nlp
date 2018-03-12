package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.ml.tensorflow.{DatasetEncoder, DatasetEncoderParams, TensorflowNer, TensorflowWrapper}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, NAMED_ENTITY, TOKEN}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.{NerTagged, TokenizedSentence}
import com.johnsnowlabs.nlp.annotators.ner.{NerApproach, Verbose}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.datasets.CoNLL
import com.johnsnowlabs.nlp.embeddings.ApproachWithWordEmbeddings
import org.apache.commons.io.IOUtils
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.slf4j.LoggerFactory
import org.tensorflow.{Graph, Session}


class NerDLApproach(override val uid: String)
  extends ApproachWithWordEmbeddings[NerDLApproach, NerDLModel]
    with HasRecursiveFit[NerDLModel]
    with NerApproach[NerDLApproach] {

  def this() = this(Identifiable.randomUID("NerDL"))

  override val description = "Trains Tensorflow based Char-CNN-BLSTM model"
  override val requiredAnnotatorTypes = Array(DOCUMENT, TOKEN)
  override val annotatorType = NAMED_ENTITY

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
    maxEpochs -> 50,
    lr -> 0.2f,
    po -> 0.05f,
    batchSize -> 9,
    dropout -> 0.5f,
    verbose -> Verbose.Silent.id
  )

  private val logger = LoggerFactory.getLogger("NerDL")

  private def getTrainDataframe(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DataFrame = {

    if (!isDefined(externalDataset))
      return dataset.toDF()

    val reader = CoNLL(3, AnnotatorType.NAMED_ENTITY)
    val dataframe = reader.readDataset($(externalDataset), dataset.sparkSession).toDF

    if (recursivePipeline.isDefined) {
      return recursivePipeline.get.transform(dataframe)
    }

    logger.warn("NER DL not in a RecursivePipeline. " +
      "It is recommended to use a com.jonsnowlabs.nlp.RecursivePipeline for " +
      "better performance during training")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setCustomBoundChars(Array("\n\n"))
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

    pipeline.fit(dataframe).transform(dataframe)
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NerDLModel = {
    require(isDefined(sourceEmbeddingsPath), "embeddings must be set before training")

    val rows = getTrainDataframe(dataset, recursivePipeline)

    val trainDataset: Array[(TextSentenceLabels, TokenizedSentence)] = NerTagged.collectTrainingInstances(rows, getInputCols, $(labelColumn))

    val labels = trainDataset.flatMap(r => r._1.labels).distinct
    val chars = trainDataset.flatMap(r => r._2.tokens.flatMap(token => token.toCharArray)).distinct

    val settings = DatasetEncoderParams(labels.toList, chars.toList)
    val encoder = new DatasetEncoder(
      embeddings.get.getEmbeddings,
      settings
    )

    val graph = new Graph()
    val session = new Session(graph)


    val graphStream = getClass.getResourceAsStream("/ner_dl/char_cnn_blstm_10_100_100_25_30.pb")
    val graphBytesDef = IOUtils.toByteArray(graphStream)
    graph.importGraphDef(graphBytesDef)

    val tf = new TensorflowWrapper(session, graph)

    val ner = try {
      val model = new TensorflowNer(tf, encoder, $(batchSize), Verbose($(verbose)))
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
  }
}

object NerDLApproach extends DefaultParamsReadable[NerDLApproach]
