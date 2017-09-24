package com.jsl.nlp.annotators

import com.jsl.nlp.util.io.ResourceHelper
import com.jsl.nlp.util.regex.MatchStrategy.MatchStrategy
import com.jsl.nlp.util.regex.{MatchStrategy, RegexRule, RuleFactory, TransformStrategy}
import com.jsl.nlp.{Annotation, AnnotatorModel, AnnotatorType, DocumentAssembler}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Matches regular expressions and maps them to specified values optionally provided
  * Rules are provided from external source file
  * @param uid internal element required for storing annotator to disk
  * @@ rules: Set of rules to be mattched
  * @@ strategy:
  *   -- MATCH_ALL brings one-to-many results
  *   -- MATCH_FIRST catches only first match
  *   -- MATCH_COMPLETE returns only if match is entire target.
  */
class RegexMatcher(override val uid: String) extends AnnotatorModel[RegexMatcher] {

  import com.jsl.nlp.AnnotatorType._

  // ToDo: Check wether this annotator can be stored to disk as is. otherwise turn regex into string
  val rules: Param[String] = new Param(this, "rules", "regex patterns to match")

  private var loadedRules: Array[(String, String)] = ResourceHelper.retrieveRegexMatchRules()

  val strategy: Param[String] = new Param(this, "strategy", "MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")

  private val matchFactory = RuleFactory.lateMatching(TransformStrategy.NO_TRANSFORM)(_)

  override val annotatorType: AnnotatorType = REGEX

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  setDefault(inputCols, Array(DOCUMENT))

  def this() = this(Identifiable.randomUID("REGEX_MATCHER"))

  def getRules: Array[(String, String)] = loadedRules

  def setRules(path: String): this.type = {
    loadedRules = ResourceHelper.retrieveRegexMatchRules(path)
    set(rules, path)
  }

  def setStrategy(value: String): this.type = set(strategy, value)

  def getStrategy: String = $(strategy).toString

  private def getFactoryStrategy: MatchStrategy = $(strategy) match {
    case "MATCH_ALL" => MatchStrategy.MATCH_ALL
    case "MATCH_FIRST" => MatchStrategy.MATCH_FIRST
    case "MATCH_COMPLETE" => MatchStrategy.MATCH_COMPLETE
    case _ => throw new IllegalArgumentException("Invalid strategy. must be MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")
  }

  /** one-to-many annotation that returns matches as annotations*/
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.flatMap { annotation =>
      matchFactory(getFactoryStrategy)
        .setRules(loadedRules.map(r => new RegexRule(r._1, r._2)))
        .findMatch(annotation.metadata(AnnotatorType.DOCUMENT)).map { m =>
          Annotation(
            annotatorType,
            m.content.start,
            m.content.end - 1,
            Map(m.identifier -> m.content.matched)
          )
        }
    }
  }
}
object RegexMatcher extends DefaultParamsReadable[RegexMatcher]