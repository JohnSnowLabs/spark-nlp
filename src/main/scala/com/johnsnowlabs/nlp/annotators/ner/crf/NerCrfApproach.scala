package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.ml.crf.{CrfParams, LinearChainCrf, TextSentenceLabels}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, NAMED_ENTITY, POS, TOKEN}
import com.johnsnowlabs.nlp.annotator.PerceptronModel
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.common.NerTagged
import com.johnsnowlabs.nlp.annotators.ner.{NerApproach, Verbose}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.datasets.CoNLL
import com.johnsnowlabs.nlp.embeddings.ApproachWithWordEmbeddings
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.nlp.{AnnotatorType, DocumentAssembler, HasRecursiveFit}
import org.apache.spark.ml.param.{DoubleParam, IntParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.slf4j.LoggerFactory

/*
  Algorithm for training Named Entity Recognition Model.
   */
class NerCrfApproach(override val uid: String)
  extends ApproachWithWordEmbeddings[NerCrfApproach, NerCrfModel]
    with HasRecursiveFit[NerCrfModel]
    with NerApproach[NerCrfApproach]
{

  def this() = this(Identifiable.randomUID("NER"))

  private val logger = LoggerFactory.getLogger("NerCrfApproach")

  override val description = "CRF based Named Entity Recognition Tagger"
  override val requiredAnnotatorTypes = Array(DOCUMENT, TOKEN, POS)
  override val annotatorType = NAMED_ENTITY

  val l2 = new DoubleParam(this, "l2", "L2 regularization coefficient")
  val c0 = new IntParam(this, "c0", "c0 params defining decay speed for gradient")
  val lossEps = new DoubleParam(this, "lossEps", "If Epoch relative improvement less than eps then training is stopped")
  val minW = new DoubleParam(this, "minW", "Features with less weights then this param value will be filtered")

  val externalFeatures = new ExternalResourceParam(this, "externalFeatures", "Additional dictionaries to use as a features")

  def setL2(l2: Double) = set(this.l2, l2)
  def setC0(c0: Int) = set(this.c0, c0)
  def setLossEps(eps: Double) = set(this.lossEps, eps)
  def setMinW(w: Double) = set(this.minW, w)

  def setExternalFeatures(value: ExternalResource) = {
    require(value.options.contains("delimiter"), "external features is a delimited text. needs 'delimiter' in options")
    set(externalFeatures, value)
  }

  def setExternalFeatures(path: String,
                          delimiter: String,
                          readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                          options: Map[String, String] = Map("format" -> "text")): this.type =
    set(externalFeatures, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))



  setDefault(
    minEpochs -> 0,
    maxEpochs -> 1000,
    l2 -> 1f,
    c0 -> 2250000,
    lossEps -> 1e-3f,
    verbose -> Verbose.Silent.id
  )


  private def getTrainDataframe(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DataFrame = {

    if (!isDefined(externalDataset))
      return dataset.toDF()

    val reader = CoNLL(3, AnnotatorType.NAMED_ENTITY)
    val dataframe = reader.readDataset($(externalDataset), dataset.sparkSession).toDF

    if (recursivePipeline.isDefined) {
      return recursivePipeline.get.transform(dataframe)
    }

    logger.warn("NER CRF not in a RecursivePipeline. " +
      "It is recommended to use a com.jonsnowlabs.nlp.RecursivePipeline for " +
      "better performance during training")
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setCustomBoundChars(Array(System.lineSeparator+System.lineSeparator))
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val posTagger = PerceptronModel.pretrained()
      .setInputCols("token", "document")
      .setOutputCol("pos")

    val pipeline = new Pipeline().setStages(
      Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger)
    )

    pipeline.fit(dataframe).transform(dataframe)
  }


  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NerCrfModel = {

    val rows = getTrainDataframe(dataset, recursivePipeline)

    val trainDataset: Array[(TextSentenceLabels, PosTaggedSentence)] = NerTagged.collectTrainingInstancesWithPos(rows, getInputCols, $(labelColumn))

    val extraFeatures = get(externalFeatures)
    val dictFeatures = DictionaryFeatures.read(extraFeatures)
    val crfDataset = FeatureGenerator(dictFeatures, embeddings)
      .generateDataset(trainDataset)

    val params = CrfParams(
      minEpochs = getOrDefault(minEpochs),
      maxEpochs = getOrDefault(maxEpochs),

      l2 = getOrDefault(l2).toFloat,
      c0 = getOrDefault(c0),
      lossEps = getOrDefault(lossEps).toFloat,

      verbose = Verbose.Epochs,
      randomSeed = get(randomSeed)
    )

    val crf = new LinearChainCrf(params)
    val crfModel = crf.trainSGD(crfDataset)

    var model = new NerCrfModel()
      .setModel(crfModel)
      .setDictionaryFeatures(dictFeatures)

    if (isDefined(entities))
      model.setEntities($(entities))

    if (isDefined(minW))
      model = model.shrink($(minW).toFloat)

    model
  }
}

object NerCrfApproach extends DefaultParamsReadable[NerCrfApproach]