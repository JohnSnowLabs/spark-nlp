package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by saif on 28/04/17.
  */
class Lemmatizer(override val uid: String) extends Annotator {

  val lemmaDict: Param[Map[String, String]] = new Param(this, "lemma dictionary", "provide a lemma dictionary")

  override val aType: String = Lemmatizer.aType

  override var requiredAnnotationTypes: Array[String] = Array(RegexTokenizer.aType)

  def this() = this(Identifiable.randomUID(Lemmatizer.aType))

  def getLemmaDict: Map[String, String] = $(lemmaDict)

  def setLemmaDict(dictionary: Map[String, String]): this.type = set(lemmaDict, dictionary)

  /**
    * Would need to verify this implementation, as I am flattening multiple to one annotations
    * @param document
    * @param annotations
    * @return
    */
  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.collect {
      case tokenAnnotation: Annotation if tokenAnnotation.aType == RegexTokenizer.aType =>
        val token = document.text.substring(tokenAnnotation.begin, tokenAnnotation.end)
        Annotation(
          aType,
          tokenAnnotation.begin,
          tokenAnnotation.end,
          Map(token -> getLemmaDict.getOrElse(token, token))
        )
    }
  }

}

object Lemmatizer extends DefaultParamsReadable[Lemmatizer] {
  val aType = "lemma"
}
