package com.jsl.nlp.annotators

import com.jsl.nlp.annotators.common.WritableAnnotatorComponent
import com.jsl.nlp.annotators.param.{AnnotatorParam, SerializedAnnotatorComponent}
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by saif on 28/04/17.
  */

class Lemmatizer(override val uid: String) extends Annotator {

  protected case class SerializedDictionary(dict: Map[String, String]) extends SerializedAnnotatorComponent[LemmatizerDictionary] {
    override def deserialize: LemmatizerDictionary = {
      LemmatizerDictionary(dict)
    }
  }

  protected case class LemmatizerDictionary(dict: Map[String, String]) extends WritableAnnotatorComponent {
    override def serialize: SerializedAnnotatorComponent[LemmatizerDictionary] =
      SerializedDictionary(dict)
  }

  val lemmaDict: AnnotatorParam[LemmatizerDictionary, SerializedDictionary] =
    new AnnotatorParam[LemmatizerDictionary, SerializedDictionary](this, "lemma dictionary", "provide a lemma dictionary")

  override val aType: String = Lemmatizer.aType

  override var requiredAnnotationTypes: Array[String] = Array(RegexTokenizer.aType)

  def this() = this(Identifiable.randomUID(Lemmatizer.aType))

  def getLemmaDict: Map[String, String] = $(lemmaDict).dict

  def setLemmaDict(dictionary: Map[String, String]): this.type = set(lemmaDict, LemmatizerDictionary(dictionary))

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
