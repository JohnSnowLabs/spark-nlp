package com.johnsnowlabs.nlp.annotators.ner

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN, NAMED_ENTITY, NAMED_ENTITY_SPAN}
import com.johnsnowlabs.nlp.annotators.common.NerTagged
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, ParamsAndFeaturesReadable}
import org.apache.spark.ml.util.Identifiable

import scala.collection.Map


/**
  * Converts IOB or IOB2 representation of NER to user-friendly.
  * See https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
  */
class NerConverter(override val uid: String) extends AnnotatorModel[NerConverter] {

  def this() = this(Identifiable.randomUID("NER_CONVERTER"))

  override protected def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = NerTagged.unpack(annotations)
    val doc = annotations.filter(a => a.annotatorType == AnnotatorType.DOCUMENT).head.result
    val entities = NerTagsEncoding.fromIOB(sentences, doc)

    entities.map{entity =>
      Annotation(AnnotatorType.NAMED_ENTITY_SPAN, entity.start, entity.end, entity.entity, Map("text" -> entity.text))
    }
  }

  override val requiredAnnotatorTypes = Array(DOCUMENT, TOKEN, NAMED_ENTITY)

  override val annotatorType: AnnotatorType = NAMED_ENTITY_SPAN
}

object NerConverter extends ParamsAndFeaturesReadable[NerConverter]
