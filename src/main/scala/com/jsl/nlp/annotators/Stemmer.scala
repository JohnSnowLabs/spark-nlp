package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Annotator, Document}
import opennlp.tools.stemmer.PorterStemmer

/**
  * Created by alext on 10/23/16.
  */
class Stemmer extends Annotator {

  override val aType: String = Stemmer.aType

  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] =
    annotations.collect {
      case token: Annotation if token.aType == RegexTokenizer.aType =>
        val stem = Stemmer.stemmer.stem(document.text.substring(token.begin, token.end))
        Annotation(aType, token.begin, token.end, Map(aType -> stem))
    }

  override val requiredAnnotationTypes = Array(RegexTokenizer.aType)
}

object Stemmer {
  val aType = "stem"
  private val stemmer = new PorterStemmer()
}