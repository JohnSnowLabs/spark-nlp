package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.ml.crf.{CrfParams, LinearChainCrf}
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.NerTagged
import com.johnsnowlabs.nlp.annotators.ner.{NerApproach, Verbose}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.Dataset

/*
  Algorithm for training Named Entity Recognition Model.
   */
class NerCrfApproach(override val uid: String)
  extends AnnotatorApproach[NerCrfModel]
    with NerApproach[NerCrfApproach] {

  def this() = this(Identifiable.randomUID("NER"))

  override val description = "CRF based Named Entity Recognition Tagger"
  override val inputAnnotatorTypes = Array(DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS)
  override val outputAnnotatorType = NAMED_ENTITY

  val l2 = new DoubleParam(this, "l2", "L2 regularization coefficient")
  val c0 = new IntParam(this, "c0", "c0 params defining decay speed for gradient")
  val lossEps = new DoubleParam(this, "lossEps", "If Epoch relative improvement less than eps then training is stopped")
  val minW = new DoubleParam(this, "minW", "Features with less weights then this param value will be filtered")
  val includeConfidence = new BooleanParam(this, "includeConfidence", "whether or not to calculate prediction confidence by token, includes in metadata")
  val externalFeatures = new ExternalResourceParam(this, "externalFeatures", "Additional dictionaries to use as a features")

  def setL2(l2: Double): this.type = set(this.l2, l2)
  def setC0(c0: Int): this.type = set(this.c0, c0)
  def setLossEps(eps: Double): this.type = set(this.lossEps, eps)
  def setMinW(w: Double): this.type = set(this.minW, w)
  def setIncludeConfidence(c: Boolean): this.type = set(includeConfidence, c)

  def getL2: Double = $(l2)
  def getC0: Int = $(c0)
  def getLossEps: Double = $(lossEps)
  def getMinW: Double = $(minW)
  def getIncludeConfidence: Boolean = $(includeConfidence)

  def setExternalFeatures(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "external features is a delimited text. needs 'delimiter' in options")
    set(externalFeatures, value)
  }

  def setExternalFeatures(path: String,
                          delimiter: String,
                          readAs: ReadAs.Format = ReadAs.TEXT,
                          options: Map[String, String] = Map("format" -> "text")): this.type =
    set(externalFeatures, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))

  setDefault(
    minEpochs -> 0,
    maxEpochs -> 1000,
    l2 -> 1f,
    c0 -> 2250000,
    lossEps -> 1e-3f,
    verbose -> Verbose.Silent.id,
    includeConfidence -> false
  )


  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NerCrfModel = {

    val rows = dataset.toDF()

    val trainDataset =
      NerTagged.collectTrainingInstancesWithPos(rows, getInputCols, $(labelColumn))

    val extraFeatures = get(externalFeatures)
    val dictFeatures = DictionaryFeatures.read(extraFeatures)
    val crfDataset = FeatureGenerator(dictFeatures)
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

    val embeddingsRef = HasStorageRef.getStorageRefFromInput(dataset, $(inputCols), AnnotatorType.WORD_EMBEDDINGS)

    val crf = new LinearChainCrf(params)
    val crfModel = crf.trainSGD(crfDataset)

    var model = new NerCrfModel()
      .setModel(crfModel)
      .setDictionaryFeatures(dictFeatures)
      .setIncludeConfidence($(includeConfidence))
      .setStorageRef(embeddingsRef)

    if (isDefined(entities))
      model.setEntities($(entities))

    if (isDefined(minW))
      model = model.shrink($(minW).toFloat)

    model
  }
}

object NerCrfApproach extends DefaultParamsReadable[NerCrfApproach]