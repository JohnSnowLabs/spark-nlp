package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset


/**
  * Uses a reference file to match a set of regular expressions and put them inside a provided key. File must be comma separated.
  *
  * Matches regular expressions and maps them to specified values optionally provided
  *
  * Rules are provided from external source file
  *
  * @param uid internal element required for storing annotator to disk
  * @@ rules: Set of rules to be mattched
  * @@ strategy:
  *
  *    -- MATCH_ALL brings one-to-many results
  *
  *    -- MATCH_FIRST catches only first match
  *
  *    -- MATCH_COMPLETE returns only if match is entire target.
  *
  *
  *    See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/RegexMatcherTestSpec.scala]] for example on how to use this API.
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  **/
class RegexMatcher(override val uid: String) extends AnnotatorApproach[RegexMatcherModel] {

  /** Matches described regex rules that come in tuples in a text file */
  override val description: String = "Matches described regex rules that come in tuples in a text file"

  /** Input annotator type: CHUNK
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = CHUNK
  /** Input annotator type: DOCUMENT
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** external resource to rules, needs 'delimiter' in options
    *
    * @group param
    **/
  val rules: ExternalResourceParam = new ExternalResourceParam(this, "externalRules", "external resource to rules, needs 'delimiter' in options")
  /** MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE
    *
    * @group param
    **/
  val strategy: Param[String] = new Param(this, "strategy", "MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")

  setDefault(
    inputCols -> Array(DOCUMENT),
    strategy -> "MATCH_ALL"
  )

  def this() = this(Identifiable.randomUID("REGEX_MATCHER"))

  /** Path to file containing a set of regex,key pair. readAs can be LINE_BY_LINE or SPARK_DATASET. options contain option passed to spark reader if readAs is SPARK_DATASET.
    *
    * @group setParam
    **/
  def setRules(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "RegexMatcher requires 'delimiter' option to be set in ExternalResource")
    set(rules, value)
  }


  /** Path to file containing a set of regex,key pair. readAs can be LINE_BY_LINE or SPARK_DATASET. options contain option passed to spark reader if readAs is SPARK_DATASET.
    *
    * @group setParam
    **/
  def setRules(path: String,
               delimiter: String,
               readAs: ReadAs.Format = ReadAs.TEXT,
               options: Map[String, String] = Map("format" -> "text")): this.type =
    set(rules, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))


  /** Can be any of MATCH_FIRST|MATCH_ALL|MATCH_COMPLETE
    *
    * @group setParam
    **/
  def setStrategy(value: String): this.type = {
    require(Seq("MATCH_ALL", "MATCH_FIRST", "MATCH_COMPLETE").contains(value.toUpperCase), "Must be MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")
    set(strategy, value.toUpperCase)
  }

  /** Can be any of MATCH_FIRST|MATCH_ALL|MATCH_COMPLETE
    *
    * @group getParam
    **/
  def getStrategy: String = $(strategy).toString

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): RegexMatcherModel = {
    val processedRules = ResourceHelper.parseTupleText($(rules))
    new RegexMatcherModel()
      .setRules(processedRules)
      .setStrategy($(strategy))
  }

}

object RegexMatcher extends DefaultParamsReadable[RegexMatcher]
