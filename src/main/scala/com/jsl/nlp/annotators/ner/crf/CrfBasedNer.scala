package com.jsl.nlp.annotators.ner.crf

import com.jsl.ml.crf.{LinearChainCrf, TrainParams, Verbose}
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

  val labelsColumn = new Param[String](this, "labelsColumn", "Column with label per each token")
  val entities = new Param[Set[String]](this, "entities", "Entities to recognize")
  val crfParams = new Param[TrainParams](this, "params", "Crf Hyper Params")

  def setLabelColumn(column: String): CrfBasedNer = set(labelsColumn, column)
  def setEntities(tags: Set[String]): CrfBasedNer = set(entities, tags)
  def setParams(params: TrainParams): CrfBasedNer = set(crfParams, params)

  override def train(dataset: Dataset[_]): CrfBasedNerModel = {

    val rows = dataset.toDF()

    val trainDataset = NerTagged.collectTrainingInstances(rows, getInputCols, $(labelsColumn))
    val crfDataset = FeatureGenerator.generateDataset(trainDataset.toIterator)

    val params = if (isDefined(crfParams))
      $(crfParams)
    else TrainParams(
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
