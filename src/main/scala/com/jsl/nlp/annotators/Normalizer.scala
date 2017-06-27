package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by alext on 10/23/16.
  */
class Normalizer(override val uid: String) extends Annotator {
  override val aType: String = Normalizer.aType

  override var requiredAnnotationTypes = Array(Stemmer.aType)

  def this() = this(Identifiable.randomUID(Normalizer.aType))

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
    }.filter(_.metadata(Normalizer.aType).nonEmpty)

}
object Normalizer extends DefaultParamsReadable[Normalizer] {
  val aType: String = "ntoken"
}