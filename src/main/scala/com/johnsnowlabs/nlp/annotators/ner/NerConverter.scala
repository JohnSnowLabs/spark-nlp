package com.johnsnowlabs.nlp.annotators.ner

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, NAMED_ENTITY, TOKEN}
import com.johnsnowlabs.nlp.annotators.common.NerTagged
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, ParamsAndFeaturesReadable}
import org.apache.spark.ml.param.StringArrayParam
import org.apache.spark.ml.util.Identifiable

import scala.collection.Map


/**
  * Converts IOB or IOB2 representation of NER to user-friendly.
  * See https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
  */
class NerConverter(override val uid: String) extends AnnotatorModel[NerConverter] {

  def this() = this(Identifiable.randomUID("NER_CONVERTER"))

  override val inputAnnotatorTypes = Array(DOCUMENT, TOKEN, NAMED_ENTITY)

  override val outputAnnotatorType: AnnotatorType = CHUNK

  val whiteList: StringArrayParam = new StringArrayParam(this, "whiteList", "If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix on labels")

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = NerTagged.unpack(annotations)
    val docs = annotations.filter(a => a.annotatorType == AnnotatorType.DOCUMENT)
    val entities = sentences.zip(docs.zipWithIndex).flatMap{case (sentence, doc) => NerTagsEncoding.fromIOB(sentence, doc._1, sentenceIndex=doc._2)}

    entities.filter(entity => get(whiteList).forall(validEntity => validEntity.contains(entity.entity))).
      zipWithIndex.map{case (entity, idx) =>
      Annotation(
        outputAnnotatorType,
        entity.start,
        entity.end,
        entity.text,
        Map("entity" -> entity.entity, "sentence" -> entity.sentenceId, "chunk" -> idx.toString)
      )
    }
  }

}

object NerConverter extends ParamsAndFeaturesReadable[NerConverter]
