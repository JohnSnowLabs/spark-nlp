package com.jsl.nlp.annotators

import com.jsl.nlp.util.{RegexRule, RegexStrategy}
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.param.Param

/**
  * Created by Saif Addin on 5/7/2017.
  */
class RegexMatcher extends Annotator {

  override val aType: String = RegexMatcher.aType

  val patterns: Param[Seq[RegexRule]] = new Param(this, "patterns", "regex patterns to match")

  def setPatterns(value: Seq[RegexRule]): RegexMatcher = set(patterns, value)

  def addPattern(value: RegexRule): RegexMatcher = set(patterns, $(patterns) :+ value)

  def getPatterns: Seq[RegexRule] = $(patterns)

  override def annotate(
                       document: Document, annotations: Seq[Annotation]
                       ): Seq[Annotation] = {
    getPatterns.flatMap(pattern => {
      pattern.strategy match {
        // MatchAll => puts all matches in the same value separated by comma. May need to change separator.
        case RegexStrategy.MatchAll => (pattern.regex findAllMatchIn document.text).map(m =>
          Some(Annotation(RegexMatcher.aType, m.start, m.end, Map(pattern.value -> m.matched)))
        )
        // MatchFirst => puts first match
        case RegexStrategy.MatchFirst => Seq((pattern.regex findFirstMatchIn document.text).map(m =>
          Annotation(RegexMatcher.aType, m.start, m.end, Map(pattern.value -> m.matched))
        ))
        // MatchComplete => puts match only if all match equals the target. may use true or false instead of text.
        case RegexStrategy.MatchComplete => {
          Seq((pattern.regex findFirstMatchIn document.text).map(m =>
            Annotation(RegexMatcher.aType, m.start, m.end, Map(pattern.value -> m.matched))
          ).filter(_.metadata(pattern.value) == document.text))
        }
      }
    }).flatten
  }

}

object RegexMatcher {
  val aType = "regex"
}