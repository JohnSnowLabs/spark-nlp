package com.jsl.nlp.annotators

import com.jsl.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Annotator that cleans out tokens. Requires stems, hence tokens
  * @param uid required internal uid for saving annotator
  */
class Normalizer(override val uid: String) extends AnnotatorModel[Normalizer] {

  import com.jsl.nlp.AnnotatorType._

  override val annotatorType: AnnotatorType = TOKEN

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("NORMALIZER"))

  /** ToDo: Review imeplementation, Current implementation generates spaces between non-words, potentially breaking tokens*/
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] =
    annotations.map { token =>
      val nToken = token.metadata(TOKEN)
        .toLowerCase
        .replaceAll("[^a-zA-Z]", "")
        .trim
      Annotation(annotatorType, token.begin, token.end, Map(annotatorType -> nToken, "sentence" -> token.metadata("sentence")))
    }.filter(_.metadata(annotatorType).nonEmpty)

}
object Normalizer extends DefaultParamsReadable[Normalizer]