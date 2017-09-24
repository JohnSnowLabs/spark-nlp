package com.jsl.nlp.annotators

import com.jsl.nlp.annotators.common.StringMapParam
import com.jsl.nlp.util.io.ResourceHelper
import com.jsl.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.collection.JavaConverters._

/**
  * Created by saif on 28/04/17.
  */

/**
  * Class to find standarized lemmas from words. Uses a user-provided or default dictionary.
  * @param uid required internal uid provided by constructor
  * @@ lemmaDict: A dictionary of predefined lemmas must be provided
  */
class Lemmatizer(override val uid: String) extends AnnotatorModel[Lemmatizer] {

  import com.jsl.nlp.AnnotatorType._

  val lemmaDict: StringMapParam = new StringMapParam(this, "lemmaDict", "provide a lemma dictionary")

  override val annotatorType: AnnotatorType = TOKEN

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("LEMMATIZER"))

  def getLemmaDict: Map[String, String] = $(lemmaDict)

  def setLemmaDict(dictionary: String): this.type = {
    set(lemmaDict, ResourceHelper.retrieveLemmaDict(dictionary))
  }

  def setLemmaDictHMap(dictionary: java.util.HashMap[String, String]): this.type = {
    set(lemmaDict, dictionary.asScala.toMap)
  }

  /**
    * @return one to one annotation from token to a lemmatized word, if found on dictionary or leave the word as is
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.collect {
      case tokenAnnotation: Annotation if tokenAnnotation.annotatorType == annotatorType =>
        val token = tokenAnnotation.metadata(annotatorType)
        Annotation(
          annotatorType,
          tokenAnnotation.begin,
          tokenAnnotation.end,
          Map[String, String](annotatorType -> $(lemmaDict).getOrElse(token, token))
        )
    }
  }

}

object Lemmatizer extends DefaultParamsReadable[Lemmatizer]
