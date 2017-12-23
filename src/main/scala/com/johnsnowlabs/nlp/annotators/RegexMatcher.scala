package com.johnsnowlabs.nlp.annotators

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
class RegexMatcher(override val uid: String) extends AnnotatorModel[RegexMatcher] {
  import com.johnsnowlabs.nlp.AnnotatorType._

  private val config: Config = ConfigFactory.load

  // ToDo: Check whether this annotator can be stored to disk as is. otherwise turn regex into string

  val rulesPath: Param[String] = new Param(this, "rulesPath", "File containing rules separated by commas")

  val rulesFormat: Param[String] = new Param(this, "rulesFormat", "TXT or TXTDS for reading as dataset")

  val rulesSeparator: Param[String] = new Param(this, "rulesSeparator", "Separator for regex rules and match")

  if (config.getString("nlp.regexMatcher.file").nonEmpty)
    setDefault(rulesPath, config.getString("nlp.regexMatcher.file"))

  setDefault(rulesFormat, config.getString("nlp.regexMatcher.format"))

  setDefault(rulesSeparator, config.getString("nlp.regexMatcher.separator"))

  val rules: Param[Array[(String, String)]] = new Param(this, "rules", "Array of rule strings separated by commas")

  val strategy: Param[String] = new Param(this, "strategy", "MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")

  def setRulesPath(path: String): this.type = set(rulesPath, path)

  def getRulesPath: String = $(rulesPath)

  def setRulesFormat(format: String): this.type = set(rulesFormat, format)

  def getRulesFormat: String = $(rulesFormat)

  def setRulesSeparator(separator: String): this.type = set(rulesSeparator, separator)

  def getRulesSeparator: String = $(rulesSeparator)

  def setRules(value: Array[(String, String)]): this.type = set(rules, value)

  def getRules: Array[(String, String)] = $(rules)

  private val matchFactory = RuleFactory.lateMatching(TransformStrategy.NO_TRANSFORM)(_)

  override val annotatorType: AnnotatorType = REGEX

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  setDefault(inputCols, Array(DOCUMENT))

  def this() = this(Identifiable.randomUID("REGEX_MATCHER"))

  def setStrategy(value: String): this.type = set(strategy, value)

  def getStrategy: String = $(strategy).toString

  private def resolveRulesFromPath(): Array[(String, String)] =
    RegexMatcher.retrieveRegexMatchRules($(rulesPath), $(rulesFormat), $(rulesSeparator))

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
        .setRules(get(rules).getOrElse(resolveRulesFromPath()).map(r => new RegexRule(r._1, r._2)))
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

object RegexMatcher extends DefaultParamsReadable[RegexMatcher] {

  /**
    * Regex matcher rules
    */
  protected def retrieveRegexMatchRules(
                               rulesFilePath: String,
                               rulesFormat: String,
                               rulesSeparator: String
                             ): Array[(String, String)] = {
    ResourceHelper.parseTupleText(rulesFilePath, rulesFormat.toUpperCase, rulesSeparator)
  }

}