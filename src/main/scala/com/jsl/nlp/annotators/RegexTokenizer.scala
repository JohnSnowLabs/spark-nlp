package com.jsl.nlp.annotators

import org.apache.spark.ml.param.Param

import com.jsl.nlp.{Document, Annotation, Annotator}

import scala.util.matching.Regex

/**
  * Created by alext on 10/23/16.
  */
class RegexTokenizer() extends Annotator {
  override val aType: String = RegexTokenizer.aType

  override val requiredAnnotationTypes: Seq[String] = Seq()

  lazy val regex: Regex = $(pattern).r

  val pattern: Param[String] = new Param(this, "pattern", "this is the token pattern")

  def setPattern(value: String): RegexTokenizer = set(pattern, value)

  def getPattern: String = $(pattern)
  setDefault(pattern, "\\w+")

  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] = regex.findAllMatchIn(document.text).map {
    m =>
      Annotation(aType, m.start, m.end)
  }.toSeq
}
object RegexTokenizer{
  val aType = "token"
}