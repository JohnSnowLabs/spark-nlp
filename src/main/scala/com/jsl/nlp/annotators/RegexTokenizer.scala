package com.jsl.nlp.annotators

import org.apache.spark.ml.param.Param
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.util.matching.Regex

/**
  * Created by alext on 10/23/16.
  */

/**
  * Tokenizes raw text into word pieces, tokens.
  * @param uid required uid for storing annotator to disk
  * @@ pattern: RegexPattern to split phrases into tokens
  */
class RegexTokenizer(override val uid: String) extends Annotator {

  val pattern: Param[String] = new Param(this, "pattern", "this is the token pattern")

  lazy val regex: Regex = $(pattern).r

  override val annotatorType: String = RegexTokenizer.annotatorType

  override var requiredAnnotatorTypes: Array[String] = Array()

  def this() = this(Identifiable.randomUID(RegexTokenizer.annotatorType))

  def setPattern(value: String): this.type = set(pattern, value)

  def getPattern: String = $(pattern)
  setDefault(pattern, "\\w+")

  /** one to many annotation */
  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] = regex.findAllMatchIn(document.text).map {
    m =>
      Annotation(annotatorType, m.start, m.end, Map(RegexTokenizer.annotatorType -> m.matched))
  }.toSeq
}
object RegexTokenizer extends DefaultParamsReadable[RegexTokenizer]{
  val annotatorType = "token"
}