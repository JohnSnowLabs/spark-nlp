package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, REGEX}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class RegexMatcher(override val uid: String) extends AnnotatorApproach[RegexMatcherModel] {

  override val description: String = "Matches described regex rules that come in tuples in a text file"

  override val annotatorType: AnnotatorType = REGEX

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  val rules: ExternalResourceParam = new ExternalResourceParam(this, "externalRules", "external resource to rules, needs 'delimiter' in options")

  val strategy: Param[String] = new Param(this, "strategy", "MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")

  setDefault(
    inputCols -> Array(DOCUMENT),
    strategy -> "MATCH_ALL"
  )

  def this() = this(Identifiable.randomUID("REGEX_MATCHER"))

  def setRules(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "RegexMatcher requires 'delimiter' option to be set in ExternalResource")
    set(rules, value)
  }

  def setRules(path: String,
               delimiter: String,
               readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
               options: Map[String, String] = Map("format" -> "text")): this.type =
    set(rules, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))

  def setStrategy(value: String): this.type = {
    require(Seq("MATCH_ALL", "MATCH_FIRST", "MATCH_COMPLETE").contains(value.toUpperCase), "Must be MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")
    set(strategy, value.toUpperCase)
  }

  def getStrategy: String = $(strategy).toString

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): RegexMatcherModel = {
    val processedRules = ResourceHelper.parseTupleText($(rules))
    new RegexMatcherModel()
      .setRules(processedRules)
      .setStrategy($(strategy))
  }

}

object RegexMatcher extends DefaultParamsReadable[RegexMatcher]
