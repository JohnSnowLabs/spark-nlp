/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/** Uses rules to match a set of regular expressions and associate them with a provided
  * identifier.
  *
  * A rule consists of a regex pattern and an identifier, delimited by a character of choice. An
  * example could be `\d{4}\/\d\d\/\d\d,date` which will match strings like `"1970/01/01"` to the
  * identifier `"date"`.
  *
  * Rules must be provided by either `setRules` (followed by `setDelimiter`) or an external file.
  *
  * To use an external file, a dictionary of predefined regular expressions must be provided with
  * `setExternalRules`. The dictionary can be set in either in the form of a delimited text file
  * or directly as an [[com.johnsnowlabs.nlp.util.io.ExternalResource ExternalResource]].
  *
  * Pretrained pipelines are available for this module, see
  * [[https://nlp.johnsnowlabs.com/docs/en/pipelines Pipelines]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/annotation/text/english/regex-matcher/Matching_Text_with_RegexMatcher.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/RegexMatcherTestSpec.scala RegexMatcherTestSpec]].
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
  *   .setExternalRules("src/test/resources/regex-matcher/rules.txt",  ",")
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
  * @param uid
  *   internal element required for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
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
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class RegexMatcher(override val uid: String) extends AnnotatorApproach[RegexMatcherModel] {

  /** Matches described regex rules that come in tuples in a text file */
  override val description: String =
    "Matches described regex rules that come in tuples in a text file"

  /** Input annotator type: CHUNK
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = CHUNK

  /** Input annotator type: DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Rules with regex pattern and identifiers for matching
    * @group param
    */
  val rules: StringArrayParam =
    new StringArrayParam(this, "rules", "Rules with regex pattern and identifiers for matching")

  /** Delimiter for rules provided with setRules
    *
    * @group param
    */
  val delimiter: Param[String] = new Param[String](this, "delimiter", "Delimiter for the rules")

  /** External resource to rules, needs 'delimiter' in options
    *
    * @group param
    */
  val externalRules: ExternalResourceParam = new ExternalResourceParam(
    this,
    "externalRules",
    "External resource to rules, needs 'delimiter' in options")

  /** Strategy for which to match the expressions (Default: `"MATCH_ALL"`). Possible values are:
    *   - MATCH_ALL brings one-to-many results
    *   - MATCH_FIRST catches only first match
    *   - MATCH_COMPLETE returns only if match is entire target.
    *
    * @group param
    */
  val strategy: Param[String] = new Param(
    this,
    "strategy",
    "Strategy for which to match the expressions (MATCH_ALL, MATCH_FIRST, MATCH_COMPLETE")

  setDefault(inputCols -> Array(DOCUMENT), strategy -> "MATCH_ALL")

  def this() = this(Identifiable.randomUID("REGEX_MATCHER"))

  /** External dictionary already in the form of [[ExternalResource]], for which the Map member
    * `options` has `"delimiter"` defined.
    *
    * Note that only either externalRules or rules can be set at once.
    *
    * ==Example==
    * {{{
    * val regexMatcher = new RegexMatcher()
    *   .setExternalRules(ExternalResource(
    *     "src/test/resources/regex-matcher/rules.txt",
    *     ReadAs.TEXT,
    *     Map("delimiter" -> ",")
    *   ))
    *   .setInputCols("sentence")
    *   .setOutputCol("regex")
    *   .setStrategy(strategy)
    * }}}
    *
    * @group setParam
    */
  def setExternalRules(value: ExternalResource): this.type = {
    require(
      value.options.contains("delimiter"),
      "RegexMatcher requires 'delimiter' option to be set in ExternalResource")
    require(get(rules).isEmpty, "Only either parameter externalRules or rules should be set.")
    require(
      get(this.delimiter).isEmpty,
      "Parameter delimiter should only be set with parameter rules. " +
        "Please provide the delimiter in the ExternalResource.")
    set(externalRules, value)
  }

  /** External dictionary to be used by the lemmatizer, which needs `delimiter` set for parsing
    * the resource.
    *
    * Note that only either externalRules or rules can be set at once.
    *
    * @group setParam
    */
  def setExternalRules(
      path: String,
      delimiter: String,
      readAs: ReadAs.Format = ReadAs.TEXT,
      options: Map[String, String] = Map("format" -> "text")): this.type = {
    require(get(rules).isEmpty, "Only either parameter externalRules or rules should be set.")
    require(
      get(this.delimiter).isEmpty,
      "Parameter delimiter should only be set with parameter rules.")
    set(externalRules, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))
  }

  /** Strategy for which to match the expressions (Default: `"MATCH_ALL"`)
    *
    * @group setParam
    */
  def setStrategy(value: String): this.type = {
    require(
      Seq("MATCH_ALL", "MATCH_FIRST", "MATCH_COMPLETE").contains(value.toUpperCase),
      "Must be MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")
    set(strategy, value.toUpperCase)
  }

  /** Strategy for which to match the expressions (Default: `"MATCH_ALL"`)
    *
    * @group getParam
    */
  def getStrategy: String = $(strategy)

  /** Sets the regex rules to match the identifier with.
    *
    * The rules must consist of a regex pattern and an identifier for that pattern. The regex
    * pattern and the identifier must be delimited by a character that will also have to set with
    * `setDelimiter`.
    *
    * Only one of either parameter `rules` or `externalRules` must be set.
    *
    * ==Example==
    * {{{
    * val regexMatcher = new RegexMatcher()
    *   .setRules(Array("\d{4}\/\d\d\/\d\d,date", "\d{2}\/\d\d\/\d\d,date_short")
    *   .setDelimiter(",")
    *   .setInputCols("sentence")
    *   .setOutputCol("regex")
    *   .setStrategy("MATCH_ALL")
    * }}}
    *
    * @group setParam
    * @param value
    *   Array of rules
    */
  def setRules(value: Array[String]): this.type = {
    require(
      get(externalRules).isEmpty,
      "Only either parameter rules or externalRules should be set.")
    set(rules, value)
  }

  /** Sets the regex rules to match the identifier with.
    *
    * Note that only either externalRules or rules can be set at once.
    *
    * @group setParam
    * @param value
    *   Array of rules and identifiers as tuples
    */
  def setDelimiter(value: String): this.type = {
    require(
      get(externalRules).isEmpty,
      "Only either parameter rules or externalRules should be set.")
    set(delimiter, value)
  }

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): RegexMatcherModel = {
    val processedRules: Array[(String, String)] =
      if (get(externalRules).nonEmpty) ResourceHelper.parseTupleText($(externalRules))
      else {
        val delim = getOrDefault(delimiter)
        getOrDefault(rules).map { rule =>
          rule.split(delim) match {
            case Array(pattern, identifier) => (pattern, identifier)
            case a: Array[String] =>
              throw new IllegalArgumentException(
                s"Expected 2-tuple after splitting, but got ${a.length} for '$rule'")
          }
        }
      }

    new RegexMatcherModel()
      .setExternalRules(processedRules)
      .setStrategy($(strategy))
  }
}

/** This is the companion object of [[RegexMatcher]]. Please refer to that class for the
  * documentation.
  */
object RegexMatcher extends DefaultParamsReadable[RegexMatcher]
