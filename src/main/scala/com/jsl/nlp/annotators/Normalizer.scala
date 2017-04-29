package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Annotator, Document}

/**
  * Created by alext on 10/23/16.
  */
class Normalizer extends Annotator {
  override val aType: String = "ntoken"

  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] =
    annotations.collect {
      case token: Annotation if token.aType == "stem" =>
        val nToken = document.text.substring(token.begin, token.end)
          .toLowerCase
          .replaceAll("[^a-zA-Z]", " ")
          .trim
        Annotation(aType, token.begin, token.end, Map(aType -> nToken))
    }.filter(_.metadata("ntoken").nonEmpty)

  override val requiredAnnotationTypes = Seq("stem")
}
