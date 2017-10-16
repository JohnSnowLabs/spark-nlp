package com.johnsnowlabs.nlp.annotators.ner.regex

import com.johnsnowlabs.nlp.annotators.common.StringMapParam
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetectorModel
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
 * Created by alext on 6/14/17.
 */
//Many Similarities with POS tagger. need a common/generic annotator
class NERRegexModel(override val uid: String) extends AnnotatorModel[NERRegexModel] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val annotatorType: AnnotatorType = NAMED_ENTITY

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  val language: Param[String] = new Param(this, "language", "this is the language of the text")

  val model = new StringMapParam(this, "entity_dictionary", "entity NER dictionary")

  lazy val entityDictionary: EntityDictionary = new EntityDictionary($(model))

  def tag(sentences: Array[String]): Seq[Map[String, String]] = {
    sentences.flatMap(entityDictionary.predict)
  }

  def setModel(value: Map[String, String]): this.type = set(model, value)

  def this() = this(Identifiable.randomUID("NER"))

  /**
   * This takes a document and annotations and produces new annotations of this annotator's annotation type
   *
   * @return
   */
  override protected def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations
      .filter(_.annotatorType == DOCUMENT)
      .flatMap( annotation => {
        annotation.result match {
          case sentence: String =>
            tag(Array(sentence)).flatMap { tag =>
              Some(Annotation(
                annotatorType,
                tag("entity"),
                Map(tag("word") -> tag("entity"), Annotation.BEGIN -> (tag("start").toInt + annotation.metadata(Annotation.BEGIN).toInt).toString,
                  Annotation.END -> (tag("end").toInt + annotation.metadata(Annotation.BEGIN)).toString)
              ))
            }
          case _ => None
        }
      })
  }

  def setLanguage(value: String): NERRegexModel = set(language, value)

  def getLanguage: String = $(language)

  setDefault(language, "person")
}

object NERRegexModel extends DefaultParamsReadable[NERRegexModel]
