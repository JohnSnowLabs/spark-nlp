package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Annotator, Document}
import opennlp.tools.stemmer.PorterStemmer

/**
  * Created by alext on 10/23/16.
  */
class Stemmer() extends Annotator {
  override val aType: String = "stem"

  override def annotate(
    document: Document, annos: Seq[Annotation]
  ): Seq[Annotation] =
    annos.collect {
      case token: Annotation if token.aType == "token" =>
        val stem = Stemmer.stemmer.stem(document.text.substring(token.begin, token.end))
        Annotation(aType, token.begin, token.end, Map(aType -> stem))
    }

  override val requiredAnnotationTypes = Seq("token")
}

object Stemmer {
  private val stemmer = new PorterStemmer()
}