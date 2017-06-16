package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Annotator, Document}
import opennlp.tools.stemmer.PorterStemmer

/**
  * Created by alext on 10/23/16.
  */
class Stemmer extends Annotator {

  override val aType: String = Stemmer.aType

  override val requiredAnnotationTypes = Array(RegexTokenizer.aType)

  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] =
    annotations.collect {
      case tokenAnnotation: Annotation if tokenAnnotation.aType == RegexTokenizer.aType =>
        val token = document.text.substring(tokenAnnotation.begin, tokenAnnotation.end)
        val stem = Stemmer.stemmer.stem(token)
        Annotation(aType, tokenAnnotation.begin, tokenAnnotation.end, Map(token -> stem))
    }

}

object Stemmer {
  val aType = "stem"
  private val stemmer = new PorterStemmer()
}