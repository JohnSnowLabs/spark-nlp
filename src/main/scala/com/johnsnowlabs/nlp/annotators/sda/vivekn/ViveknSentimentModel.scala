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

package com.johnsnowlabs.nlp.annotators.sda.vivekn

import com.johnsnowlabs.nlp.annotators.common.{TokenizedSentence, TokenizedWithSentence}
import com.johnsnowlabs.nlp.serialization.{MapFeature, SetFeature}
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.param.{DoubleParam, IntParam, LongParam}
import org.apache.spark.ml.util.Identifiable

/** Sentiment analyser inspired by the algorithm by Vivek Narayanan
  * [[https://github.com/vivekn/sentiment/]].
  *
  * The algorithm is based on the paper
  * [[https://arxiv.org/abs/1305.6143 "Fast and accurate sentiment classification using an enhanced Naive Bayes model"]].
  *
  * This is the instantiated model of the
  * [[com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach ViveknSentimentApproach]].
  * For training your own model, please see the documentation of that class.
  *
  * The analyzer requires sentence boundaries to give a score in context. Tokenization is needed
  * to make sure tokens are within bounds. Transitivity requirements are also required.
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn ViveknSentimentTestSpec]].
  *
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector SentimentDetector]] for an
  *   alternative approach to sentiment detection
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
class ViveknSentimentModel(override val uid: String)
    extends AnnotatorModel[ViveknSentimentModel]
    with HasSimpleAnnotate[ViveknSentimentModel]
    with ViveknSentimentUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Output annotator type : SENTIMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = SENTIMENT

  /** Input annotator type : SENTIMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  /** positive_sentences
    *
    * @group param
    */
  protected val positive: MapFeature[String, Long] = new MapFeature(this, "positive_sentences")

  /** negative_sentences
    *
    * @group param
    */
  protected val negative: MapFeature[String, Long] = new MapFeature(this, "negative_sentences")

  /** words
    *
    * @group param
    */
  protected val words: SetFeature[String] = new SetFeature[String](this, "words")

  /** Count of positive words
    *
    * @group param
    */
  val positiveTotals: LongParam =
    new LongParam(this, "positive_totals", "Count of positive words")

  /** Count of negative words
    *
    * @group param
    */
  val negativeTotals: LongParam =
    new LongParam(this, "negative_totals", "Count of negative words")

  /** Proportion of feature content to be considered relevant (Default: `0.5`)
    *
    * @group param
    */
  val importantFeatureRatio = new DoubleParam(
    this,
    "importantFeatureRatio",
    "Proportion of feature content to be considered relevant (Default: `0.5`)")

  /** Proportion to lookahead in unimportant features (Default: `0.025`)
    *
    * @group param
    */
  val unimportantFeatureStep = new DoubleParam(
    this,
    "unimportantFeatureStep",
    "Proportion to lookahead in unimportant features (Default: `0.025`)")

  /** Content feature limit, to boost performance in very dirt text (Default: disabled with `-1`)
    *
    * @group param
    */
  val featureLimit = new IntParam(
    this,
    "featureLimit",
    "Content feature limit, to boost performance in very dirt text (Default disabled with: -1")

  def this() = this(Identifiable.randomUID("VIVEKN"))

  /** Set Proportion of feature content to be considered relevant (Default: `0.5`)
    *
    * @group setParam
    */
  def setImportantFeatureRatio(v: Double): this.type = set(importantFeatureRatio, v)

  /** Set Proportion to lookahead in unimportant features (Default: `0.025`)
    *
    * @group setParam
    */
  def setUnimportantFeatureStep(v: Double): this.type = set(unimportantFeatureStep, v)

  /** Set Content feature limit, to boost performance in very dirt text (Default: disabled with
    * `-1`)
    *
    * @group setParam
    */
  def setFeatureLimit(v: Int): this.type = set(featureLimit, v)

  /** Get Proportion of feature content to be considered relevant (Default: `0.5`) */
  def getImportantFeatureRatio(v: Double): Double = $(importantFeatureRatio)

  /** Get Proportion to lookahead in unimportant features (Default: `0.025`) */
  def getUnimportantFeatureStep(v: Double): Double = $(unimportantFeatureStep)

  /** Get Content feature limit, to boost performance in very dirt text (Default: disabled with
    * `-1`)
    *
    * @group getParam
    */
  def getFeatureLimit(v: Int): Int = $(featureLimit)

  /** Count of positive words
    *
    * @group getParam
    */
  def getPositive: Map[String, Long] = $$(positive)

  /** Count of negative words
    *
    * @group getParam
    */
  def getNegative: Map[String, Long] = $$(negative)

  /** Set of unique words
    *
    * @group getParam
    */
  def getFeatures: Set[String] = $$(words)

  private[vivekn] def setPositive(value: Map[String, Long]): this.type = set(positive, value)

  private[vivekn] def setNegative(value: Map[String, Long]): this.type = set(negative, value)

  private[vivekn] def setPositiveTotals(value: Long): this.type = set(positiveTotals, value)

  private[vivekn] def setNegativeTotals(value: Long): this.type = set(negativeTotals, value)

  private[vivekn] def setWords(value: Array[String]): this.type = {
    require(
      value.nonEmpty,
      "Word analysis for features cannot be empty. Set prune to false if training is small")
    val currentFeatures = scala.collection.mutable.Set.empty[String]
    val start = (value.length * $(importantFeatureRatio)).ceil.toInt
    val afterStart = {
      if ($(featureLimit) == -1) value.length
      else $(featureLimit)
    }
    val step = (afterStart * $(unimportantFeatureStep)).ceil.toInt
    value.take(start).foreach(currentFeatures.add)
    Range(start, afterStart, step).foreach(k => {
      value.slice(k, k + step).foreach(currentFeatures.add)
    })

    set(words, currentFeatures.toSet)
  }

  /** Positive: 0, Negative: 1, NA: 2 */
  def classify(sentence: TokenizedSentence): (Short, Double) = {
    val wordFeatures = negateSequence(sentence.tokens).intersect($$(words)).toList
    if (wordFeatures.isEmpty) return (2, 0.0)
    val positiveScore = wordFeatures
      .map(word =>
        scala.math.log(($$(positive).getOrElse(word, 0L) + 1.0) / (2.0 * $(positiveTotals))))
      .sum
    val negativeScore = wordFeatures
      .map(word =>
        scala.math.log(($$(negative).getOrElse(word, 0L) + 1.0) / (2.0 * $(negativeTotals))))
      .sum
    val positiveSum = wordFeatures.map(word => $$(positive).getOrElse(word, 0L)).sum.toDouble
    val negativeSum = wordFeatures.map(word => $$(negative).getOrElse(word, 0L)).sum.toDouble
    lazy val positiveConfidence = positiveSum / (positiveSum + negativeSum)
    lazy val negativeConfidence = negativeSum / (positiveSum + negativeSum)
    if (positiveScore > negativeScore) (0, positiveConfidence) else (1, negativeConfidence)
  }

  /** Tokens are needed to identify each word in a sentence boundary POS tags are optionally
    * submitted to the model in case they are needed Lemmas are another optional annotator for
    * some models Bounds of sentiment are hardcoded to 0 as they render useless
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = TokenizedWithSentence.unpack(annotations)

    sentences
      .filter(s => s.indexedTokens.nonEmpty)
      .map(sentence => {
        val (result, confidence) = classify(sentence)
        Annotation(
          outputAnnotatorType,
          sentence.indexedTokens.map(t => t.begin).min,
          sentence.indexedTokens.map(t => t.end).max,
          if (result == 0) "positive" else if (result == 1) "negative" else "na",
          Map("confidence" -> confidence.toString.take(6)))
      })
  }

}

trait ReadablePretrainedVivekn
    extends ParamsAndFeaturesReadable[ViveknSentimentModel]
    with HasPretrained[ViveknSentimentModel] {
  override val defaultModelName = Some("sentiment_vivekn")

  /** Java compliant-overrides */
  override def pretrained(): ViveknSentimentModel = super.pretrained()
  override def pretrained(name: String): ViveknSentimentModel = super.pretrained(name)
  override def pretrained(name: String, lang: String): ViveknSentimentModel =
    super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): ViveknSentimentModel =
    super.pretrained(name, lang, remoteLoc)
}

/** This is the companion object of [[ViveknSentimentModel]]. Please refer to that class for the
  * documentation.
  */
object ViveknSentimentModel extends ReadablePretrainedVivekn
