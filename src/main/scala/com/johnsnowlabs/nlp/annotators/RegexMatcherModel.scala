package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, REGEX}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.util.regex.MatchStrategy.MatchStrategy
import com.johnsnowlabs.nlp.util.regex.{MatchStrategy, RegexRule, RuleFactory, TransformStrategy}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, DocumentAssembler}
import com.typesafe.config.{Config, ConfigFactory}
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
class RegexMatcherModel(override val uid: String) extends AnnotatorModel[RegexMatcherModel] {

  override val annotatorType: AnnotatorType = REGEX

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  val rules: Param[Array[(String, String)]] = new Param(this, "rules", "Array of rule strings separated by commas")

  val strategy: Param[String] = new Param(this, "strategy", "MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")

  def this() = this(Identifiable.randomUID("REGEX_MATCHER"))

  def setStrategy(value: String): this.type = set(strategy, value)

  def getStrategy: String = $(strategy).toString

  def setRules(value: Array[(String, String)]): this.type = set(rules, value)

  def getRules: Array[(String, String)] = $(rules)

  private def getFactoryStrategy: MatchStrategy = $(strategy) match {
    case "MATCH_ALL" => MatchStrategy.MATCH_ALL
    case "MATCH_FIRST" => MatchStrategy.MATCH_FIRST
    case "MATCH_COMPLETE" => MatchStrategy.MATCH_COMPLETE
    case _ => throw new IllegalArgumentException("Invalid strategy. must be MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")
  }

  lazy private val matchFactory = RuleFactory
    .lateMatching(TransformStrategy.NO_TRANSFORM)(getFactoryStrategy)
    .setRules($(rules).map(r => new RegexRule(r._1, r._2)))

  /** one-to-many annotation that returns matches as annotations*/
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.flatMap { annotation =>
      matchFactory
        .findMatch(annotation.result).map { m =>
          Annotation(
            annotatorType,
            m.content.start,
            m.content.end - 1,
            m.content.matched,
            Map.empty[String, String]
          )
        }
    }
  }
}

object RegexMatcherModel extends DefaultParamsReadable[RegexMatcherModel]