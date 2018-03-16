package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.param.{BooleanParam, Param}
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
  val lowercase = new BooleanParam(this, "lowercase", "whether to convert strings to lowercase")

  setDefault(pattern, "[^\\pL+]")
  setDefault(lowercase, true)

  def getPattern: String = $(pattern)

  def setPattern(value: String): this.type = set(pattern, value)

  def getLowercase: Boolean = $(lowercase)

  def setLowercase(value: Boolean): this.type = set(lowercase, value)

  def this() = this(Identifiable.randomUID("NORMALIZER"))

  /** ToDo: Review implementation, Current implementation generates spaces between non-words, potentially breaking tokens*/
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] =
    annotations.map { token =>
      val cased =
        if ($(lowercase)) token.result.toLowerCase
        else token.result

      val nToken = cased
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