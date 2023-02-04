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
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, IntParam, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/** Annotator that cleans out tokens. Requires stems, hence tokens. Removes all dirty characters
  * from text following a regex pattern and transforms words based on a provided dictionary
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/document-normalizer/document_normalizer_notebook.ipynb Examples]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.{Normalizer, Tokenizer}
  * import org.apache.spark.ml.Pipeline
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val normalizer = new Normalizer()
  *   .setInputCols("token")
  *   .setOutputCol("normalized")
  *   .setLowercase(true)
  *   .setCleanupPatterns(Array("""[^\w\d\s]""")) // remove punctuations (keep alphanumeric chars)
  * // if we don't set CleanupPatterns, it will only keep alphabet letters ([^A-Za-z])
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   normalizer
  * ))
  *
  * val data = Seq("John and Peter are brothers. However they don't support each other that much.")
  *   .toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("normalized.result").show(truncate = false)
  * +----------------------------------------------------------------------------------------+
  * |result                                                                                  |
  * +----------------------------------------------------------------------------------------+
  * |[john, and, peter, are, brothers, however, they, dont, support, each, other, that, much]|
  * +----------------------------------------------------------------------------------------+
  * }}}
  * @param uid
  *   required internal uid for saving annotator
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
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
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class Normalizer(override val uid: String) extends AnnotatorApproach[NormalizerModel] {

  /** Cleans out tokens */
  override val description: String = "Cleans out tokens"

  /** Output Annotator Type : TOKEN
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** Input Annotator Type : TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  /** Normalization regex patterns which match will be removed from token (Default:
    * `Array("[^\\pL+]")`)
    *
    * @group param
    */
  val cleanupPatterns = new StringArrayParam(
    this,
    "cleanupPatterns",
    "Normalization regex patterns which match will be removed from token")

  /** Normalization regex patterns which match will be removed from token (Default:
    * `Array("[^\\pL+]")`)
    * @group getParam
    */
  def getCleanupPatterns: Array[String] = $(cleanupPatterns)

  /** Normalization regex patterns which match will be removed from token (Default:
    * `Array("[^\\pL+]")`)
    * @group setParam
    */
  def setCleanupPatterns(value: Array[String]): this.type = set(cleanupPatterns, value)

  /** Whether to convert strings to lowercase (Default: `false`)
    *
    * @group param
    */
  val lowercase = new BooleanParam(
    this,
    "lowercase",
    "Whether to convert strings to lowercase (Default: `false`)")

  /** Whether to convert strings to lowercase (Default: `false`)
    * @group getParam
    */
  def getLowercase: Boolean = $(lowercase)

  /** Whether to convert strings to lowercase (Default: `false`)
    * @group setParam
    */
  def setLowercase(value: Boolean): this.type = set(lowercase, value)

  /** Delimited file with list of custom words to be manually corrected
    *
    * @group param
    */
  val slangDictionary = new ExternalResourceParam(
    this,
    "slangDictionary",
    "Delimited file with list of custom words to be manually corrected")

  /** Delimited file with list of custom words to be manually corrected
    * @group setParam
    */
  def setSlangDictionary(value: ExternalResource): this.type = {
    require(
      value.options.contains("delimiter"),
      "slang dictionary is a delimited text. needs 'delimiter' in options")
    set(slangDictionary, value)
  }

  /** Delimited file with list of custom words to be manually corrected
    * @group setParam
    */
  def setSlangDictionary(
      path: String,
      delimiter: String,
      readAs: ReadAs.Format = ReadAs.TEXT,
      options: Map[String, String] = Map("format" -> "text")): this.type =
    set(slangDictionary, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))

  /** Whether or not to be case sensitive to match slangs (Default: `false`)
    *
    * @group param
    */
  val slangMatchCase = new BooleanParam(
    this,
    "slangMatchCase",
    "Whether or not to be case sensitive to match slangs. Defaults to false.")

  /** Whether or not to be case sensitive to match slangs (Default: `false`)
    * @group setParam
    */
  def setSlangMatchCase(value: Boolean): this.type = set(slangMatchCase, value)

  /** Whether or not to be case sensitive to match slangs (Default: `false`)
    * @group getParam
    */
  def getSlangMatchCase: Boolean = $(slangMatchCase)

  /** Set the minimum allowed length for each token (Default: `0`)
    *
    * @group param
    */
  val minLength = new IntParam(this, "minLength", "Set the minimum allowed length for each token")

  /** Set the minimum allowed length for each token (Default: `0`)
    * @group setParam
    */
  def setMinLength(value: Int): this.type = {
    require(value >= 0, "minLength must be greater equal than 0")
    require(value.isValidInt, "minLength must be Int")
    set(minLength, value)
  }

  /** Set the minimum allowed length for each token (Default: `0`)
    * @group getParam
    */
  def getMinLength: Int = $(minLength)

  /** Set the maximum allowed length for each token
    *
    * @group param
    */
  val maxLength = new IntParam(this, "maxLength", "Set the maximum allowed length for each token")

  /** Set the maximum allowed length for each token
    * @group setParam
    */
  def setMaxLength(value: Int): this.type = {
    require(
      value >= $ {
        minLength
      },
      "maxLength must be greater equal than minLength")
    require(value.isValidInt, "minLength must be Int")
    set(maxLength, value)
  }

  /** Set the maximum allowed length for each token
    * @group getParam
    */
  def getMaxLength: Int = $(maxLength)

  setDefault(
    lowercase -> false,
    cleanupPatterns -> Array("[^\\pL+]"),
    slangMatchCase -> false,
    minLength -> 0)

  def this() = this(Identifiable.randomUID("NORMALIZER"))

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): NormalizerModel = {

    val loadSlangs = if (get(slangDictionary).isDefined) {
      val parsed = ResourceHelper.parseKeyValueText($(slangDictionary))
      if ($(slangMatchCase))
        parsed.mapValues(_.trim)
      else
        parsed.map { case (k, v) => (k.toLowerCase, v.trim) }
    } else
      Map.empty[String, String]

    val raw = new NormalizerModel()
      .setCleanupPatterns($(cleanupPatterns))
      .setLowercase($(lowercase))
      .setSlangDict(loadSlangs)
      .setSlangMatchCase($(slangMatchCase))
      .setMinLength($(minLength))

    if (isDefined(maxLength))
      raw.setMaxLength($(maxLength))

    raw
  }

}

/** This is the companion object of [[Normalizer]]. Please refer to that class for the
  * documentation.
  */
object Normalizer extends DefaultParamsReadable[Normalizer]
