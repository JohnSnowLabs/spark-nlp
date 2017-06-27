package com.jsl.nlp.annotators

import com.jsl.nlp.util.ResourceHelper
import com.jsl.nlp.{Annotation, Annotator, Document}

/**
  * Created by saif on 28/04/17.
  */
class Lemmatizer(lemmaDict: Map[String, String] = ResourceHelper.defaultLemmaDict) extends Annotator {

  override val aType: String = Lemmatizer.aType

  override val requiredAnnotationTypes: Array[String] = Array(RegexTokenizer.aType)

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
          Map(token -> lemmaDict.getOrElse(token, token))
        )
    }
  }

}

object Lemmatizer {
  val aType = "lemma"
}
