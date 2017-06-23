package com.jsl.nlp.annotators

import org.apache.spark.ml.param.Param
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.util.matching.Regex

/**
  * Created by alext on 10/23/16.
  */
class RegexTokenizer(override val uid: String) extends Annotator {

  val pattern: Param[String] = new Param(this, "pattern", "this is the token pattern")

  lazy val regex: Regex = $(pattern).r

  override val aType: String = RegexTokenizer.aType

  override var requiredAnnotationTypes: Array[String] = Array()

  def this() = this(Identifiable.randomUID(RegexTokenizer.aType))

  def setPattern(value: String): this.type = set(pattern, value)

  def getPattern: String = $(pattern)
  setDefault(pattern, "\\w+")

  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] = regex.findAllMatchIn(document.text).map {
    m =>
      Annotation(aType, m.start, m.end, Map(RegexTokenizer.aType -> m.matched))
  }.toSeq
}
object RegexTokenizer extends DefaultParamsReadable[RegexTokenizer]{
  val aType = "token"
}