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

package com.johnsnowlabs.nlp.annotators.sda.pragmatic

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, SENTIMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, DoubleParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/** Trains a rule based sentiment detector, which calculates a score based on predefined keywords.
  *
  * A dictionary of predefined sentiment keywords must be provided with `setDictionary`, where
  * each line is a word delimited to its class (either `positive` or `negative`). The dictionary
  * can be set in either in the form of a delimited text file or directly as an
  * [[com.johnsnowlabs.nlp.util.io.ExternalResource ExternalResource]].
  *
  * By default, the sentiment score will be assigned labels `"positive"` if the score is `>= 0`,
  * else `"negative"`. To retrieve the raw sentiment scores, `enableScore` needs to be set to
  * `true`.
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/dictionary-sentiment/sentiment.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sda/pragmatic/PragmaticSentimentTestSpec.scala SentimentTestSpec]].
  *
  * ==Example==
  * In this example, the dictionary `default-sentiment-dict.txt` has the form of
  * {{{
  * ...
  * cool,positive
  * superb,positive
  * bad,negative
  * uninspired,negative
  * ...
  * }}}
  * where each sentiment keyword is delimited by `","`.
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.Lemmatizer
  * import com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector
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
  * val lemmatizer = new Lemmatizer()
  *   .setInputCols("token")
  *   .setOutputCol("lemma")
  *   .setDictionary("src/test/resources/lemma-corpus-small/lemmas_small.txt", "->", "\t")
  *
  * val sentimentDetector = new SentimentDetector()
  *   .setInputCols("lemma", "document")
  *   .setOutputCol("sentimentScore")
  *   .setDictionary("src/test/resources/sentiment-corpus/default-sentiment-dict.txt", ",", ReadAs.TEXT)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   lemmatizer,
  *   sentimentDetector,
  * ))
  *
  * val data = Seq(
  *   "The staff of the restaurant is nice",
  *   "I recommend others to avoid because it is too expensive"
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("sentimentScore.result").show(false)
  * +----------+  //  +------+ for enableScore set to true
  * |result    |  //  |result|
  * +----------+  //  +------+
  * |[positive]|  //  |[1.0] |
  * |[negative]|  //  |[-2.0]|
  * +----------+  //  +------+
  * }}}
  *
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach ViveknSentimentApproach]]
  *   for an alternative approach to sentiment extraction
  * @param uid
  *   internal uid needed for saving annotator to disk
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
class SentimentDetector(override val uid: String)
    extends AnnotatorApproach[SentimentDetectorModel] {

  /** Output annotation type : SENTIMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = SENTIMENT

  /** Input annotation type : TOKEN, DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  /** Rule based sentiment detector */
  override val description: String = "Rule based sentiment detector"

  def this() = this(Identifiable.randomUID("SENTIMENT"))

  /** Multiplier for positive sentiments (Default: `1.0`)
    *
    * @group param
    */
  val positiveMultiplier = new DoubleParam(
    this,
    "positiveMultiplier",
    "multiplier for positive sentiments. Defaults 1.0")

  /** Multiplier for negative sentiments (Default: `-1.0`)
    *
    * @group param
    */
  val negativeMultiplier = new DoubleParam(
    this,
    "negativeMultiplier",
    "multiplier for negative sentiments. Defaults -1.0")

  /** Multiplier for increment sentiments (Default: `2.0`)
    *
    * @group param
    */
  val incrementMultiplier = new DoubleParam(
    this,
    "incrementMultiplier",
    "multiplier for increment sentiments. Defaults 2.0")

  /** Multiplier for decrement sentiments (Default: `-2.0`)
    *
    * @group param
    */
  val decrementMultiplier = new DoubleParam(
    this,
    "decrementMultiplier",
    "multiplier for decrement sentiments. Defaults -2.0")

  /** Multiplier for revert sentiments (Default: `-1.0`)
    *
    * @group param
    */
  val reverseMultiplier =
    new DoubleParam(this, "reverseMultiplier", "multiplier for revert sentiments. Defaults -1.0")

  /** If true, score will show as the double value, else will output string `"positive"` or
    * `"negative"` (Default: `false`)
    *
    * @group param
    */
  val enableScore = new BooleanParam(
    this,
    "enableScore",
    "If true, score will show as the double value, else will output string \"positive\" or \"negative\". Defaults false")

  /** Delimited file with a list sentiment tags per word (either `positive` or `negative`).
    * Requires '`delimiter`' in `options`.
    * ==Example==
    * {{{
    * cool,positive
    * superb,positive
    * bad,negative
    * uninspired,negative
    * }}}
    * where the '`delimiter`' options was set with `Map("delimiter" -> ",")`
    *
    * @group param
    */
  val dictionary = new ExternalResourceParam(
    this,
    "dictionary",
    "delimited file with a list sentiment tags per word. Requires 'delimiter' in options")

  setDefault(
    positiveMultiplier -> 1.0,
    negativeMultiplier -> -1.0,
    incrementMultiplier -> 2.0,
    decrementMultiplier -> -2.0,
    reverseMultiplier -> -1.0,
    enableScore -> false)

  /** Multiplier for positive sentiments (Default: `1.0`)
    *
    * @group param
    */
  def setPositiveMultiplier(v: Double): this.type = set(positiveMultiplier, v)

  /** Multiplier for negative sentiments (Default: `-1.0`)
    *
    * @group param
    */
  def setNegativeMultiplier(v: Double): this.type = set(negativeMultiplier, v)

  /** Multiplier for increment sentiments (Default: `2.0`)
    *
    * @group param
    */
  def setIncrementMultiplier(v: Double): this.type = set(incrementMultiplier, v)

  /** Multiplier for decrement sentiments (Default: `-2.0`)
    *
    * @group param
    */
  def setDecrementMultiplier(v: Double): this.type = set(decrementMultiplier, v)

  /** Multiplier for revert sentiments (Default: `-1.0`)
    *
    * @group param
    */
  def setReverseMultiplier(v: Double): this.type = set(reverseMultiplier, v)

  /** If true, score will show as the double value, else will output string `"positive"` or
    * `"negative"` (Default: `false`)
    *
    * @group param
    */
  def setEnableScore(v: Boolean): this.type = set(enableScore, v)

  /** Delimited file with a list sentiment tags per word. Requires 'delimiter' in options.
    * Dictionary needs 'delimiter' in order to separate words from sentiment tags
    *
    * @group param
    */
  def setDictionary(value: ExternalResource): this.type = {
    require(
      value.options.contains("delimiter"),
      "dictionary needs 'delimiter' in order to separate words from sentiment tags")
    set(dictionary, value)
  }

  /** Delimited file with a list sentiment tags per word. Requires 'delimiter' in options.
    * Dictionary needs 'delimiter' in order to separate words from sentiment tags
    *
    * @group param
    */
  def setDictionary(
      path: String,
      delimiter: String,
      readAs: ReadAs.Format,
      options: Map[String, String] = Map("format" -> "text")): this.type =
    set(dictionary, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): SentimentDetectorModel = {
    new SentimentDetectorModel()
      .setIncrementMultipler($(incrementMultiplier))
      .setDecrementMultipler($(decrementMultiplier))
      .setPositiveMultipler($(positiveMultiplier))
      .setNegativeMultipler($(negativeMultiplier))
      .setReverseMultipler($(reverseMultiplier))
      .setEnableScore($(enableScore))
      .setSentimentDict(ResourceHelper.parseKeyValueText($(dictionary)))
  }

}

/** This is the companion object of [[SentimentDetector]]. Please refer to that class for the
  * documentation.
  */
object SentimentDetector extends DefaultParamsReadable[SentimentDetector]
