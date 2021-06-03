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
  * Uses a reference file to match a set of regular expressions and associate them with a provided identifier.
  *
  * A dictionary of predefined regular expressions must be provided with `setRules`.
  * The dictionary can be set in either in the form of a delimited text file or directly as an
  * [[com.johnsnowlabs.nlp.util.io.ExternalResource ExternalResource]].
  *
  * For extended examples of usage, see the [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb Spark NLP Workshop]]
  * and the [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/RegexMatcherTestSpec.scala RegexMatcherTestSpec]].
  *
  * ==Example==
  * In this example, the `rules.txt` has the form of
  * {{{
  * the\s\w+, followed by 'the'
  * ceremonies, ceremony
  * }}}
  * where each regex is separated by the identifier by `","`
  * {{{
  * import ResourceHelper.spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.SentenceDetector
  * import com.johnsnowlabs.nlp.annotators.RegexMatcher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
  *
  * val sentence = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
  *
  * val regexMatcher = new RegexMatcher()
  *   .setRules("src/test/resources/regex-matcher/rules.txt",  ",")
  *   .setInputCols(Array("sentence"))
  *   .setOutputCol("regex")
  *   .setStrategy("MATCH_ALL")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, sentence, regexMatcher))
  *
  * val data = Seq(
  *   "My first sentence with the first rule. This is my second sentence with ceremonies rule."
  * ).toDF("text")
  * val results = pipeline.fit(data).transform(data)
  *
  * results.selectExpr("explode(regex) as result").show(false)
  * +--------------------------------------------------------------------------------------------+
  * |result                                                                                      |
  * +--------------------------------------------------------------------------------------------+
  * |[chunk, 23, 31, the first, [identifier -> followed by 'the', sentence -> 0, chunk -> 0], []]|
  * |[chunk, 71, 80, ceremonies, [identifier -> ceremony, sentence -> 1, chunk -> 0], []]        |
  * +--------------------------------------------------------------------------------------------+
  * }}}
  *
  * @param uid internal element required for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio anno  1
  * @groupprio param  2
  * @groupprio setParam  3
  * @groupprio getParam  4
  * @groupprio Ungrouped 5
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

  /**
    * external resource to rules, needs 'delimiter' in options
    * @group param
    **/
  val rules: ExternalResourceParam = new ExternalResourceParam(this, "externalRules", "external resource to rules, needs 'delimiter' in options")
  /**
    * Strategy for which to match the expressions (Default: `"MATCH_ALL"`).
    * Possible values are:
    *  - MATCH_ALL brings one-to-many results
    *  - MATCH_FIRST catches only first match
    *  - MATCH_COMPLETE returns only if match is entire target.
    * @group param
    **/
  val strategy: Param[String] = new Param(this, "strategy", "Strategy for which to match the expressions (MATCH_ALL, MATCH_FIRST, MATCH_COMPLETE")

  setDefault(
    inputCols -> Array(DOCUMENT),
    strategy -> "MATCH_ALL"
  )

  def this() = this(Identifiable.randomUID("REGEX_MATCHER"))

  /**
    * External dictionary already in the form of [[ExternalResource]], for which the Map member `options`
    * has `"delimiter"` defined.
    * ==Example==
    * {{{
    * val regexMatcher = new RegexMatcher()
    *   .setRules(ExternalResource(
    *     "src/test/resources/regex-matcher/rules.txt",
    *     ReadAs.TEXT,
    *     Map("delimiter" -> ",")
    *   ))
    *   .setInputCols("sentence")
    *   .setOutputCol("regex")
    *   .setStrategy(strategy)
    * }}}
    * @group setParam
    **/
  def setRules(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "RegexMatcher requires 'delimiter' option to be set in ExternalResource")
    set(rules, value)
  }


  /**
    * External dictionary to be used by the lemmatizer, which needs `delimiter` set for parsing
    * the resource
    * @group setParam
    * */
  def setRules(path: String,
               delimiter: String,
               readAs: ReadAs.Format = ReadAs.TEXT,
               options: Map[String, String] = Map("format" -> "text")): this.type =
    set(rules, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))


  /**
    * Strategy for which to match the expressions (Default: `"MATCH_ALL"`)
    * @group setParam
    **/
  def setStrategy(value: String): this.type = {
    require(Seq("MATCH_ALL", "MATCH_FIRST", "MATCH_COMPLETE").contains(value.toUpperCase), "Must be MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")
    set(strategy, value.toUpperCase)
  }

  /**
    * Strategy for which to match the expressions (Default: `"MATCH_ALL"`)
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
