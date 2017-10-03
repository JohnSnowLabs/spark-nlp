package com.jsl.nlp.annotators.ner.linearcrf

import com.jsl.ml.crf.{LinearChainCrfModel, SerializedLinearChainCrfModel}
import com.jsl.nlp.{Annotation, AnnotatorModel}
import com.jsl.nlp.AnnotatorType._
import com.jsl.nlp.annotators.param.AnnotatorParam
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable


class CrfBasedNerModel (override val uid: String) extends AnnotatorModel[CrfBasedNerModel] {

  def this() = this(Identifiable.randomUID("NER"))

  val entities = new Param[Set[String]](this, "entities", "Set of Entities to recognize")

  val model = new AnnotatorParam[LinearChainCrfModel, SerializedLinearChainCrfModel](
    this, "CRF Model", "Trained CRF model ")

  def setModel(crf: LinearChainCrfModel): CrfBasedNerModel = set(model, crf)
  def setEntities(toExtract: Set[String]): CrfBasedNerModel = set(entities, toExtract)

  override protected def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sourceSentences = PosTagged.unpack(annotations)

    val crf = $(model)
    sourceSentences.flatMap{sentence =>
      val instance = FeatureGenerator.generate(sentence, crf.metadata)
      val labelIds = crf.predict(instance)
      sentence.indexedTaggedWords
        .zip(labelIds.labels)
        .flatMap{case (word, labelId) =>
          val label = crf.metadata.labels(labelId)

          if (!isDefined(entities) || $(entities).contains(label) || $(entities).isEmpty)
            Some(new Annotation(annotatorType, word.begin, word.end, Map("tag" -> label)))
          else
            None.asInstanceOf[Some[Annotation]]
        }
    }
  }

  override val requiredAnnotatorTypes = Array(DOCUMENT, TOKEN, POS)

  override val annotatorType: AnnotatorType = NAMED_ENTITY
}

