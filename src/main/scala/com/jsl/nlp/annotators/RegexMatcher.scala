package com.jsl.nlp.annotators

import com.jsl.nlp.util.{RegexRule, RegexStrategy}
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.param.Param

/**
  * Created by Saif Addin on 5/7/2017.
  */
class RegexMatcher extends Annotator {

  override val aType: String = RegexMatcher.aType

  override val requiredAnnotationTypes = Seq()

  private val patterns: Param[Seq[RegexRule]] = new Param(this, "patterns", "regex patterns to match")

  def setPatterns(value: Seq[RegexRule]): RegexMatcher = set(patterns, value)

  def addPattern(value: RegexRule): RegexMatcher = set(patterns, $(patterns) :+ value)

  def getPatterns: Seq[RegexRule] = $(patterns)

  override def annotate(
                       document: Document, annotations: Seq[Annotation]
                       ): Seq[Annotation] = {
    Seq(
      Annotation(
        RegexMatcher.aType,
        0,
        document.text.length + 1,
        getPatterns.map(pattern => {(
          pattern.value,
          pattern.strategy match {
              // MatchAll => puts all matches in the same value separated by comma. May need to change separator.
            case RegexStrategy.MatchAll => (pattern.regex findAllMatchIn document.text).map(m => m.matched).mkString(",")
              // MatchFirst => puts first match
            case RegexStrategy.MatchFirst => (pattern.regex findFirstMatchIn document.text).map(_.matched).getOrElse("")
              // MatchComplete => puts match only if all match equals the target. may use true or false instead of text.
            case RegexStrategy.MatchComplete => {
              if ((pattern.regex findFirstMatchIn document.text).map(_.matched).getOrElse("") == document.text)
                document.text else ""
            }
          }
          )}).toMap.filterNot(_._2.isEmpty)
      )
    )
  }

}

object RegexMatcher {
  val aType = "regex"
}