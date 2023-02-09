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

import com.johnsnowlabs.nlp.annotators.common.TokenizedWithSentence
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{
  Annotation,
  AnnotatorModel,
  HasSimpleAnnotate,
  ParamsAndFeaturesReadable
}
import org.apache.spark.ml.param.{BooleanParam, DoubleParam}
import org.apache.spark.ml.util.Identifiable

/** Rule based sentiment detector, which calculates a score based on predefined keywords.
  *
  * This is the instantiated model of the [[SentimentDetector]]. For training your own model,
  * please see the documentation of that class.
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
class SentimentDetectorModel(override val uid: String)
    extends AnnotatorModel[SentimentDetectorModel]
    with HasSimpleAnnotate[SentimentDetectorModel] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Sentiment dict
    *
    * @group param
    */
  val sentimentDict = new MapFeature[String, String](this, "sentimentDict")

  /** @group param */
  lazy val model: PragmaticScorer = new PragmaticScorer(
    $$(sentimentDict),
    $(positiveMultiplier),
    $(negativeMultiplier),
    $(incrementMultiplier),
    $(decrementMultiplier),
    $(reverseMultiplier))

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

  def this() = this(Identifiable.randomUID("SENTIMENT"))

  /** Multiplier for positive sentiments (Default: `1.0`)
    *
    * @group param
    */
  val positiveMultiplier = new DoubleParam(
    this,
    "positiveMultiplier",
    "Multiplier for positive sentiments. Defaults 1.0")

  /** Multiplier for negative sentiments (Default: `-1.0`)
    *
    * @group param
    */
  val negativeMultiplier = new DoubleParam(
    this,
    "negativeMultiplier",
    "Multiplier for negative sentiments. Defaults -1.0")

  /** Multiplier for increment sentiments (Default: `2.0`)
    *
    * @group param
    */
  val incrementMultiplier = new DoubleParam(
    this,
    "incrementMultiplier",
    "Multiplier for increment sentiments. Defaults 2.0")

  /** Multiplier for decrement sentiments (Default: `-2.0`)
    *
    * @group param
    */
  val decrementMultiplier = new DoubleParam(
    this,
    "decrementMultiplier",
    "Multiplier for decrement sentiments. Defaults -2.0")

  /** Multiplier for revert sentiments (Default: `-1.0`)
    *
    * @group param
    */
  val reverseMultiplier =
    new DoubleParam(this, "reverseMultiplier", "Multiplier for revert sentiments. Defaults -1.0")

  /** if true, score will show as a string type containing a double value, else will output string
    * `"positive"` or `"negative"` (Default: `false`)
    *
    * @group param
    */
  val enableScore = new BooleanParam(
    this,
    "enableScore",
    "if true, score will show as a string type containing a double value, else will output string \"positive\" or \"negative\". Defaults false")

  /** Multiplier for positive sentiments (Default: `1.0`)
    *
    * @group setParam
    */
  def setPositiveMultipler(v: Double): this.type = set(positiveMultiplier, v)

  /** Multiplier for negative sentiments (Default: `-1.0`)
    *
    * @group setParam
    */
  def setNegativeMultipler(v: Double): this.type = set(negativeMultiplier, v)

  /** Multiplier for increment sentiments (Default: `2.0`)
    *
    * @group setParam
    */
  def setIncrementMultipler(v: Double): this.type = set(incrementMultiplier, v)

  /** Multiplier for decrement sentiments (Default: `-2.0`)
    *
    * @group setParam
    */
  def setDecrementMultipler(v: Double): this.type = set(decrementMultiplier, v)

  /** Multiplier for revert sentiments (Default: `-1.0`)
    *
    * @group setParam
    */
  def setReverseMultipler(v: Double): this.type = set(reverseMultiplier, v)

  /** If true, score will show as a string type containing a double value, else will output string
    * `"positive"` or `"negative"` (Default: `false`)
    *
    * @group setParam
    */
  def setEnableScore(v: Boolean): this.type = set(enableScore, v)

  /** Path to file with list of inputs and their content, with such delimiter, readAs LINE_BY_LINE
    * or as SPARK_DATASET. If latter is set, options is passed to spark reader.
    *
    * @group setParam
    */
  def setSentimentDict(value: Map[String, String]): this.type = set(sentimentDict, value)

  /** Tokens are needed to identify each word in a sentence boundary POS tags are optionally
    * submitted to the model in case they are needed Lemmas are another optional annotator for
    * some models Bounds of sentiment are hardcoded to 0 as they render useless
    *
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val tokenizedSentences = TokenizedWithSentence.unpack(annotations)

    val score = model.score(tokenizedSentences.toArray)

    Seq(
      Annotation(
        outputAnnotatorType,
        0,
        0, {
          if ($(enableScore)) score.toString else if (score >= 0) "positive" else "negative"
        },
        Map.empty[String, String]))
  }

}

object SentimentDetectorModel extends ParamsAndFeaturesReadable[SentimentDetectorModel]
