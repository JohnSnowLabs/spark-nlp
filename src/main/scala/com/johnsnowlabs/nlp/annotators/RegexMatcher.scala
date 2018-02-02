package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, REGEX}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class RegexMatcher(override val uid: String) extends AnnotatorApproach[RegexMatcherModel] {

  override val description: String = "Matches described regex rules that come in tuples in a text file"

  override val annotatorType: AnnotatorType = REGEX

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  val rulesPath: Param[String] = new Param(this, "rulesPath", "File containing rules separated by commas")

  val rulesFormat: Param[String] = new Param(this, "rulesFormat", "TXT or TXTDS for reading as dataset")

  val rulesSeparator: Param[String] = new Param(this, "rulesSeparator", "Separator for regex rules and match")

  val rules: Param[Array[(String, String)]] = new Param(this, "rules", "Array of rule strings separated by commas")

  val strategy: Param[String] = new Param(this, "strategy", "MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")

  setDefault(
    inputCols -> Array(DOCUMENT),
    rulesFormat -> "TXT",
    rulesSeparator -> ",",
    strategy -> "MATCH_ALL"
  )

  def this() = this(Identifiable.randomUID("REGEX_MATCHER"))

  def setRules(value: Array[(String, String)]): this.type = set(rules, value)

  def getRules: Array[(String, String)] = $(rules)

  def setStrategy(value: String): this.type = set(strategy, value)

  def getStrategy: String = $(strategy).toString

  def setRulesPath(path: String): this.type = set(rulesPath, path)

  def getRulesPath: String = $(rulesPath)

  def setRulesFormat(format: String): this.type = set(rulesFormat, format)

  def getRulesFormat: String = $(rulesFormat)

  def setRulesSeparator(separator: String): this.type = set(rulesSeparator, separator)

  def getRulesSeparator: String = $(rulesSeparator)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): RegexMatcherModel = {
    require(get(rulesPath).isDefined || get(rules).isDefined)
    val processedRules = get(rules) ++ get(rulesPath).map(path => ResourceHelper.parseTupleText(path, $(rulesFormat), $(rulesSeparator)))
    new RegexMatcherModel()
      .setRules(processedRules.toArray.flatten)
      .setStrategy($(strategy))
  }

}

object RegexMatcher extends DefaultParamsReadable[RegexMatcher]
