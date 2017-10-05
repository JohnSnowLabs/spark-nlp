package com.jsl.nlp.annotators.ner.crf

import com.jsl.ml.crf.{LinearChainCrf, CrfParams, Verbose}
import com.jsl.nlp.AnnotatorApproach
import com.jsl.nlp.AnnotatorType.{DOCUMENT, NAMED_ENTITY, POS, TOKEN}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset


class CrfBasedNer(override val uid: String) extends AnnotatorApproach[CrfBasedNerModel]{
  def this() = this(Identifiable.randomUID("NER"))

  override val description: String = "CRF based Named Entity Recognition tagger"
  override val requiredAnnotatorTypes = Array(DOCUMENT, TOKEN, POS)
  override val annotatorType = NAMED_ENTITY

  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each token")
  val entities = new Param[Array[String]](this, "entities", "Entities to recognize")
  val crfParams = new Param[CrfParams](this, "crfParams", "Crf Hyper Params")

  def setLabelColumn(column: String): CrfBasedNer = set(labelColumn, column)
  def setEntities(tags: Array[String]): CrfBasedNer = set(entities, tags)
  def setCrfParams(params: CrfParams): CrfBasedNer = set(crfParams, params)

  override def train(dataset: Dataset[_]): CrfBasedNerModel = {

    val rows = dataset.toDF()

    val trainDataset = NerTagged.collectTrainingInstances(rows, getInputCols, $(labelColumn))
    val crfDataset = FeatureGenerator.generateDataset(trainDataset.toIterator)

    val params = if (isDefined(crfParams))
      $(crfParams)
    else CrfParams(
      l2 = 1f,
      verbose = Verbose.Epochs,
      randomSeed = Some(0),
      c0 = 2250000
    )

    val crf = new LinearChainCrf(params)
    val crfModel = crf.trainSGD(crfDataset)

    val model = new CrfBasedNerModel()
      .setModel(crfModel)

    if (isDefined(entities))
      model.setEntities($(entities))

    model
  }
}
