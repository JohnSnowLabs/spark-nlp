package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Annotator, Document}

/**
  * Created by alext on 10/23/16.
  */
class Normalizer extends Annotator {
  override val aType: String = Normalizer.aType

  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] =
    annotations.collect {
      case token: Annotation if token.aType == Stemmer.aType =>
        val nToken = document.text.substring(token.begin, token.end)
          .toLowerCase
          .replaceAll("[^a-zA-Z]", " ")
          .trim
        Annotation(aType, token.begin, token.end, Map(aType -> nToken))
    }.filter(_.metadata(Normalizer.token).nonEmpty)

  override val requiredAnnotationTypes = Seq(Stemmer.aType)
}
object Normalizer {
  val aType: String = "ntoken"
  val token = "ntoken"
}