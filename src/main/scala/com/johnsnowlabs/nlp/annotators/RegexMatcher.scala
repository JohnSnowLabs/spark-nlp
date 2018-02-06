package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, REGEX}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class RegexMatcher(override val uid: String) extends AnnotatorApproach[RegexMatcherModel] {

  override val description: String = "Matches described regex rules that come in tuples in a text file"

  override val annotatorType: AnnotatorType = REGEX

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  val externalRules: ExternalResourceParam = new ExternalResourceParam(this, "externalRules", "external resource to rules, needs 'delimiter' in options")

  val rules: Param[Array[(String, String)]] = new Param(this, "rules", "Array of rule strings separated by commas")

  val strategy: Param[String] = new Param(this, "strategy", "MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")

  setDefault(
    inputCols -> Array(DOCUMENT),
    strategy -> "MATCH_ALL"
  )

  def this() = this(Identifiable.randomUID("REGEX_MATCHER"))

  def setExternalRules(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "RegexMatcher requires 'delimiter' option to be set in ExternalResource")
    set(externalRules, value)
  }

  def setRules(value: Array[(String, String)]): this.type = set(rules, value)

  def getRules: Array[(String, String)] = $(rules)

  def setStrategy(value: String): this.type = set(strategy, value)

  def getStrategy: String = $(strategy).toString

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): RegexMatcherModel = {
    require(get(externalRules).isDefined || get(rules).isDefined)
    val processedRules = get(rules) ++ get(externalRules).map(path => ResourceHelper.parseTupleText($(externalRules)))
    new RegexMatcherModel()
      .setRules(processedRules.toArray.flatten)
      .setStrategy($(strategy))
  }

}

object RegexMatcher extends DefaultParamsReadable[RegexMatcher]
