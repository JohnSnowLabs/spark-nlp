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

import com.johnsnowlabs.collections.SearchTrie
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, ParamsAndFeaturesWritable}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/** Annotator to match exact phrases (by token) provided in a file against a Document.
  *
  * A text file of predefined phrases must be provided with `setEntities`. The text file can als
  * be set directly as an [[com.johnsnowlabs.nlp.util.io.ExternalResource ExternalResource]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/annotation/text/english/text-matcher-pipeline/extractor.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/TextMatcherTestSpec.scala TextMatcherTestSpec]].
  *
  * ==Example==
  * In this example, the entities file is of the form
  * {{{
  * ...
  * dolore magna aliqua
  * lorem ipsum dolor. sit
  * laborum
  * ...
  * }}}
  * where each line represents an entity phrase to be extracted.
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.Tokenizer
  * import com.johnsnowlabs.nlp.annotator.TextMatcher
  * import com.johnsnowlabs.nlp.util.io.ReadAs
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val data = Seq("Hello dolore magna aliqua. Lorem ipsum dolor. sit in laborum").toDF("text")
  * val entityExtractor = new TextMatcher()
  *   .setInputCols("document", "token")
  *   .setEntities("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.TEXT)
  *   .setOutputCol("entity")
  *   .setCaseSensitive(false)
  *   .setTokenizer(tokenizer.fit(data))
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityExtractor))
  * val results = pipeline.fit(data).transform(data)
  *
  * results.selectExpr("explode(entity) as result").show(false)
  * +------------------------------------------------------------------------------------------+
  * |result                                                                                    |
  * +------------------------------------------------------------------------------------------+
  * |[chunk, 6, 24, dolore magna aliqua, [entity -> entity, sentence -> 0, chunk -> 0], []]    |
  * |[chunk, 27, 48, Lorem ipsum dolor. sit, [entity -> entity, sentence -> 0, chunk -> 1], []]|
  * |[chunk, 53, 59, laborum, [entity -> entity, sentence -> 0, chunk -> 2], []]               |
  * +------------------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.btm.BigTextMatcher BigTextMatcher]] to match large amounts
  *   of text
  * @param uid
  *   internal uid required to generate writable annotators
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
class TextMatcher(override val uid: String)
    extends AnnotatorApproach[TextMatcherModel]
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("ENTITY_EXTRACTOR"))

  /** Output annotator type : DOCUMENT, TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes = Array(DOCUMENT, TOKEN)

  /** Output annotator type : CHUNK
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = CHUNK

  /** Extracts entities from target dataset given in a text file */
  override val description: String = "Extracts entities from target dataset given in a text file"

  /** External resource for the entities, e.g. a text file where each line is the string of an
    * entity
    *
    * @group param
    */
  val entities = new ExternalResourceParam(this, "entities", "External resource for the entities")

  /** Whether to match regardless of case (Default: `true`)
    *
    * @group param
    */
  val caseSensitive =
    new BooleanParam(this, "caseSensitive", "Whether to match regardless of case. Defaults true")

  /** Whether to merge overlapping matched chunks (Default: `false`)
    *
    * @group param
    */
  val mergeOverlapping = new BooleanParam(
    this,
    "mergeOverlapping",
    "Whether to merge overlapping matched chunks. Defaults false")

  /** Value for the entity metadata field (Default: `"entity"`)
    *
    * @group param
    */
  val entityValue = new Param[String](this, "entityValue", "Value for the entity metadata field")

  /** Whether the TextMatcher should take the CHUNK from TOKEN or not (Default: `false`)
    *
    * @group param
    */
  val buildFromTokens = new BooleanParam(
    this,
    "buildFromTokens",
    "Whether the TextMatcher should take the CHUNK from TOKEN or not")

  /** The [[Tokenizer]] to perform tokenization with
    *
    * @group param
    */
  val tokenizer = new StructFeature[TokenizerModel](this, "tokenizer")

  setDefault(inputCols, Array(TOKEN))
  setDefault(caseSensitive, true)
  setDefault(mergeOverlapping, false)
  setDefault(entityValue, "entity")
  setDefault(buildFromTokens, false)

  /** Provides a file with phrases to match (Default: Looks up path in configuration)
    *
    * @group getParam
    */
  def setEntities(value: ExternalResource): this.type =
    set(entities, value)

  /** Provides a file with phrases to match. Default: Looks up path in configuration.
    *
    * @param path
    *   a path to a file that contains the entities in the specified format.
    * @param readAs
    *   the format of the file, can be one of {ReadAs.TEXT, ReadAs.SPARK}. Defaults to
    *   ReadAs.TEXT.
    * @param options
    *   a map of additional parameters. Defaults to `Map("format" -> "text")`.
    * @return
    *   this
    * @group getParam
    */
  def setEntities(
      path: String,
      readAs: ReadAs.Format,
      options: Map[String, String] = Map("format" -> "text")): this.type =
    set(entities, ExternalResource(path, readAs, options))

  /** The [[Tokenizer]] to perform tokenization with
    *
    * @group setParam
    */
  def setTokenizer(tokenizer: TokenizerModel): this.type = set(this.tokenizer, tokenizer)

  /** The [[Tokenizer]] to perform tokenization with
    *
    * @group getParam
    */
  def getTokenizer: TokenizerModel = $$(tokenizer)

  /** Whether to match regardless of case (Default: `true`)
    *
    * @group setParam
    */
  def setCaseSensitive(v: Boolean): this.type = set(caseSensitive, v)

  /** Whether to match regardless of case (Default: `true`)
    *
    * @group getParam
    */
  def getCaseSensitive: Boolean = $(caseSensitive)

  /** Whether to merge overlapping matched chunks (Default: `false`)
    *
    * @group setParam
    */
  def setMergeOverlapping(v: Boolean): this.type = set(mergeOverlapping, v)

  /** Whether to merge overlapping matched chunks (Default: `false`)
    *
    * @group getParam
    */
  def getMergeOverlapping: Boolean = $(mergeOverlapping)

  /** Setter for Value for the entity metadata field
    *
    * @group setParam
    */
  def setEntityValue(v: String): this.type = set(entityValue, v)

  /** Getter for Value for the entity metadata field
    *
    * @group getParam
    */
  def getEntityValue: String = $(entityValue)

  /** Setter for buildFromTokens param
    *
    * @group setParam
    */
  def setBuildFromTokens(v: Boolean): this.type = set(buildFromTokens, v)

  /** Getter for buildFromTokens param
    *
    * @group getParam
    */
  def getBuildFromTokens: Boolean = $(buildFromTokens)

  /** Loads entities from a provided source. */
  private def loadEntities(dataset: Dataset[_]): Array[Array[String]] = {
    val phrases: Array[String] = ResourceHelper.parseLines($(entities))
    val parsedEntities: Array[Array[String]] = {
      get(tokenizer) match {
        case Some(tokenizerModel: TokenizerModel) =>
          phrases.map { line =>
            val annotation = Seq(Annotation(line))
            val tokens = tokenizerModel.annotate(annotation)
            tokens.map(_.result).toArray
          }
        case _ =>
          phrases.map { line =>
            line.split(" ")
          }
      }
    }
    parsedEntities
  }

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): TextMatcherModel = {
    new TextMatcherModel()
      .setSearchTrie(SearchTrie.apply(loadEntities(dataset), $(caseSensitive)))
      .setMergeOverlapping($(mergeOverlapping))
      .setBuildFromTokens($(buildFromTokens))
  }

}

/** This is the companion object of [[TextMatcher]]. Please refer to that class for the
  * documentation.
  */
object TextMatcher extends DefaultParamsReadable[TextMatcher]
