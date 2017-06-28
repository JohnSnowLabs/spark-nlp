package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by alext on 10/23/16.
  */

/**
  * Annotator that cleans out tokens. Requires stems, hence tokens
  * @param uid required internal uid for saving annotator
  */
class Normalizer(override val uid: String) extends Annotator {
  override val annotatorType: String = Normalizer.annotatorType

  override var requiredAnnotatorTypes = Array(Stemmer.annotatorType)

  def this() = this(Identifiable.randomUID(Normalizer.annotatorType))

  /** ToDo: Review imeplementation, Current implementation generates spaces between non-words, potentially breaking tokens*/
  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] =
    annotations.collect {
      case token: Annotation if token.annotatorType == Stemmer.annotatorType =>
        val nToken = document.text.substring(token.begin, token.end)
          .toLowerCase
          .replaceAll("[^a-zA-Z]", " ")
          .trim
        Annotation(annotatorType, token.begin, token.end, Map(annotatorType -> nToken))
    }.filter(_.metadata(Normalizer.annotatorType).nonEmpty)

}
object Normalizer extends DefaultParamsReadable[Normalizer] {
  val annotatorType: String = "ntoken"
}