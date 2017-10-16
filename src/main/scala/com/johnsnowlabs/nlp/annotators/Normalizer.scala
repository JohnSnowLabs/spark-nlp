package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Annotator that cleans out tokens. Requires stems, hence tokens
  * @param uid required internal uid for saving annotator
  */
class Normalizer(override val uid: String) extends AnnotatorModel[Normalizer] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val annotatorType: AnnotatorType = TOKEN

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  val pattern = new Param[String](this, "pattern", "normalization regex pattern which match will be replaced with a space")

  setDefault(pattern, "[^a-zA-Z]")

  def getPattern: String = $(pattern)

  def setPattern(value: String): this.type = set(pattern, value)

  def this() = this(Identifiable.randomUID("NORMALIZER"))

  /** ToDo: Review imeplementation, Current implementation generates spaces between non-words, potentially breaking tokens*/
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] =
    annotations.map { token =>
      val nToken = token.result
        .toLowerCase
        .replaceAll($(pattern), "")
        .trim
      Annotation(
        annotatorType,
        token.begin,
        token.end,
        nToken,
        token.metadata
      )
    }.filter(_.result.nonEmpty)

}
object Normalizer extends DefaultParamsReadable[Normalizer]