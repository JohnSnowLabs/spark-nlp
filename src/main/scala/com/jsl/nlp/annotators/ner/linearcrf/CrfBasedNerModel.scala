package com.jsl.nlp.annotators.ner.linearcrf

import com.jsl.ml.crf.{LinearChainCrfModel, SerializedLinearChainCrfModel}
import com.jsl.nlp.{Annotation, AnnotatorModel}
import com.jsl.nlp.AnnotatorType._
import com.jsl.nlp.annotators.param.AnnotatorParam
import org.apache.spark.ml.util.Identifiable


class CrfBasedNerModel (override val uid: String) extends AnnotatorModel[CrfBasedNerModel] {

  def this() = this(Identifiable.randomUID("SENTENCE"))

  val model = new AnnotatorParam[LinearChainCrfModel, SerializedLinearChainCrfModel](
    this, "CRF Model", "Trained CRF model ")

  override protected def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val instances = FeatureGeneartor.toInstances()
  }

  override val requiredAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN, POS)

  override val annotatorType: AnnotatorType = NAMED_ENTITY
}

