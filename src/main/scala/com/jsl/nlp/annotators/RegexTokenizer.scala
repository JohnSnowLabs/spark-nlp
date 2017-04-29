package com.jsl.nlp.annotators

import org.apache.spark.ml.param.Param

import com.jsl.nlp.{Document, Annotation, Annotator}

import scala.util.matching.Regex

/**
  * Created by alext on 10/23/16.
  */
class RegexTokenizer() extends Annotator {
  override val aType: String = "token"

  override val requiredAnnotationTypes: Seq[String] = Seq()

  val pattern: Param[String] = new Param(this, "pattern", "this is the token pattern")

  def setPattern(value: String): RegexTokenizer = set(pattern, value)

  def getPattern: String = $(pattern)

  setDefault(pattern, "\\w+")

  lazy val regex: Regex = $(pattern).r

  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] = regex.findAllMatchIn(document.text).map {
    m =>
      Annotation(aType, m.start, m.end)
  }.toSeq
}
