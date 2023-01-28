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

import com.johnsnowlabs.nlp._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.StringArrayParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

/** Tokenizes raw text recursively based on a handful of definable rules.
  *
  * Unlike the [[Tokenizer]], the RecursiveTokenizer operates based on these array string
  * parameters only:
  *   - `prefixes`: Strings that will be split when found at the beginning of token.
  *   - `suffixes`: Strings that will be split when found at the end of token.
  *   - `infixes`: Strings that will be split when found at the middle of token.
  *   - `whitelist`: Whitelist of strings not to split
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/training/italian/Training_Context_Spell_Checker_Italian.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/TokenizerTestSpec.scala TokenizerTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.RecursiveTokenizer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new RecursiveTokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer
  * ))
  *
  * val data = Seq("One, after the Other, (and) again. PO, QAM,").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("token.result").show(false)
  * +------------------------------------------------------------------+
  * |result                                                            |
  * +------------------------------------------------------------------+
  * |[One, ,, after, the, Other, ,, (, and, ), again, ., PO, ,, QAM, ,]|
  * +------------------------------------------------------------------+
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
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
class RecursiveTokenizer(override val uid: String)
    extends AnnotatorApproach[RecursiveTokenizerModel]
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("SILLY_TOKENIZER"))

  /** Strings that will be split when found at the beginning of token (Default: `Array("'", "\"",
    * "(", "[", "\n")`).
    *
    * @group param
    */
  val prefixes = new StringArrayParam(
    this,
    "prefixes",
    "Strings that will be split when found at the beginning of token.")

  /** Strings that will be split when found at the beginning of token.
    *
    * @group setParam
    */
  def setPrefixes(p: Array[String]): this.type = set(prefixes, p.sortBy(_.size).reverse)

  /** Strings that will be split when found at the end of token (Default: `Array(".", ":", "%",
    * ",", ";", "?", "'", "\"", ")", "]", "\n", "!", "'s")`).
    *
    * @group param
    */
  val suffixes = new StringArrayParam(
    this,
    "suffixes",
    "Strings that will be split when found at the end of token.")

  /** Strings that will be split when found at the end of token.
    *
    * @group setParam
    */
  def setSuffixes(s: Array[String]): this.type = set(suffixes, s.sortBy(_.size).reverse)

  /** Strings that will be split when found at the middle of token (Default: `Array("\n", "(",
    * ")")`).
    *
    * @group param
    */
  val infixes = new StringArrayParam(
    this,
    "infixes",
    "Strings that will be split when found at the middle of token.")

  /** Strings that will be split when found at the middle of token.
    *
    * @group setParam
    */
  def setInfixes(p: Array[String]): this.type = set(infixes, p.sortBy(_.size).reverse)

  /** Whitelist (Default: `Array("it's", "that's", "there's", "he's", "she's", "what's", "let's",
    * "who's", "It's", "That's", "There's", "He's", "She's", "What's", "Let's", "Who's")`).
    *
    * @group param
    */
  val whitelist = new StringArrayParam(this, "whitelist", "Whitelist.")

  /** Whitelist.
    *
    * @group setParam
    */
  def setWhitelist(w: Array[String]): this.type = set(whitelist, w)

  setDefault(infixes, Array("\n", "(", ")"))
  setDefault(prefixes, Array("'", "\"", "(", "[", "\n"))
  setDefault(suffixes, Array(".", ":", "%", ",", ";", "?", "'", "\"", ")", "]", "\n", "!", "'s"))
  setDefault(
    whitelist,
    Array(
      "it's",
      "that's",
      "there's",
      "he's",
      "she's",
      "what's",
      "let's",
      "who's",
      "It's",
      "That's",
      "There's",
      "He's",
      "She's",
      "What's",
      "Let's",
      "Who's"))

  /** Output Annotator Type : TOKEN
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.TOKEN

  /** Input Annotator Type : DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)

  /** Simplest possible tokenizer */
  override val description: String = "Simplest possible tokenizer"

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): RecursiveTokenizerModel = {
    new RecursiveTokenizerModel()
      .setPrefixes(getOrDefault(prefixes))
      .setSuffixes(getOrDefault(suffixes))
      .setInfixes(getOrDefault(infixes))
      .setWhitelist(getOrDefault(whitelist).toSet)
  }
}
