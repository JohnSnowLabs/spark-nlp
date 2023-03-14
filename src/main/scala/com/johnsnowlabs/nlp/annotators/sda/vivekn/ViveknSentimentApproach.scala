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

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorType}
import com.johnsnowlabs.util.spark.MapAccumulator
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/** Trains a sentiment analyser inspired by the algorithm by Vivek Narayanan
  * [[https://github.com/vivekn/sentiment/]].
  *
  * The algorithm is based on the paper
  * [[https://arxiv.org/abs/1305.6143 "Fast and accurate sentiment classification using an enhanced Naive Bayes model"]].
  *
  * The analyzer requires sentence boundaries to give a score in context. Tokenization is needed
  * to make sure tokens are within bounds. Transitivity requirements are also required.
  *
  * The training data needs to consist of a column for normalized text and a label column (either
  * `"positive"` or `"negative"`).
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn ViveknSentimentTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.Normalizer
  * import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach
  * import com.johnsnowlabs.nlp.Finisher
  * import org.apache.spark.ml.Pipeline
  *
  * val document = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val token = new Tokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val normalizer = new Normalizer()
  *   .setInputCols("token")
  *   .setOutputCol("normal")
  *
  * val vivekn = new ViveknSentimentApproach()
  *   .setInputCols("document", "normal")
  *   .setSentimentCol("train_sentiment")
  *   .setOutputCol("result_sentiment")
  *
  * val finisher = new Finisher()
  *   .setInputCols("result_sentiment")
  *   .setOutputCols("final_sentiment")
  *
  * val pipeline = new Pipeline().setStages(Array(document, token, normalizer, vivekn, finisher))
  *
  * val training = Seq(
  *   ("I really liked this movie!", "positive"),
  *   ("The cast was horrible", "negative"),
  *   ("Never going to watch this again or recommend it to anyone", "negative"),
  *   ("It's a waste of time", "negative"),
  *   ("I loved the protagonist", "positive"),
  *   ("The music was really really good", "positive")
  * ).toDF("text", "train_sentiment")
  * val pipelineModel = pipeline.fit(training)
  *
  * val data = Seq(
  *   "I recommend this movie",
  *   "Dont waste your time!!!"
  * ).toDF("text")
  * val result = pipelineModel.transform(data)
  *
  * result.select("final_sentiment").show(false)
  * +---------------+
  * |final_sentiment|
  * +---------------+
  * |[positive]     |
  * |[negative]     |
  * +---------------+
  * }}}
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
class ViveknSentimentApproach(override val uid: String)
    extends AnnotatorApproach[ViveknSentimentModel]
    with ViveknSentimentUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Vivekn inspired sentiment analysis model */
  override val description: String = "Vivekn inspired sentiment analysis model"

  /** Column with the sentiment result of every row. Must be `"positive"` or `"negative"`
    *
    * @group param
    */
  val sentimentCol = new Param[String](
    this,
    "sentimentCol",
    "column with the sentiment result of every row. Must be 'positive' or 'negative'")

  /** Removes unfrequent scenarios from scope. The higher the better performance (Default: `1`)
    *
    * @group param
    */
  val pruneCorpus = new IntParam(
    this,
    "pruneCorpus",
    "Removes unfrequent scenarios from scope. The higher the better performance. Defaults 1")

  /** Proportion of feature content to be considered relevant (Default: `0.5`)
    *
    * @group param
    */
  val importantFeatureRatio = new DoubleParam(
    this,
    "importantFeatureRatio",
    "Proportion of feature content to be considered relevant. Defaults to 0.5")

  /** Proportion to lookahead in unimportant features (Default: `0.025`)
    *
    * @group param
    */
  val unimportantFeatureStep = new DoubleParam(
    this,
    "unimportantFeatureStep",
    "Proportion to lookahead in unimportant features. Defaults to 0.025")

  /** content feature limit, to boost performance in very dirt text (Default: Disabled with `-1`)
    *
    * @group param
    */
  val featureLimit = new IntParam(
    this,
    "featureLimit",
    "content feature limit, to boost performance in very dirt text. Default disabled with -1")

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

  /** Set content feature limit, to boost performance in very dirt text (Default: Disabled with
    * `-1`)
    *
    * @group setParam
    */
  def setFeatureLimit(v: Int): this.type = set(featureLimit, v)

  /** Get Proportion of feature content to be considered relevant (Default: Disabled with `0.5`)
    *
    * @group getParam
    */
  def getImportantFeatureRatio(v: Double): Double = $(importantFeatureRatio)

  /** Get Proportion to lookahead in unimportant features (Default: `0.025`)
    *
    * @group getParam
    */
  def getUnimportantFeatureStep(v: Double): Double = $(unimportantFeatureStep)

  /** Get content feature limit, to boost performance in very dirt text (Default: Disabled with
    * `-1`)
    *
    * @group getParam
    */
  def getFeatureLimit(v: Int): Int = $(featureLimit)

  setDefault(
    importantFeatureRatio -> 0.5,
    unimportantFeatureStep -> 0.025,
    featureLimit -> -1,
    pruneCorpus -> 1)

  def this() = this(Identifiable.randomUID("VIVEKN"))

  /** Output annotator type : SENTIMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = SENTIMENT

  /** Input annotator type : TOKEN, DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  /** Column with sentiment analysis row’s result for training. If not set, external sources need
    * to be set instead. Column with the sentiment result of every row. Must be 'positive' or
    * 'negative'
    *
    * @group setParam
    */
  def setSentimentCol(value: String): this.type = set(sentimentCol, value)

  /** when training on small data you may want to disable this to not cut off infrequent words
    *
    * @group setParam
    */
  def setPruneCorpus(value: Int): this.type = set(pruneCorpus, value)

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): ViveknSentimentModel = {

    require(
      get(sentimentCol).isDefined,
      "ViveknSentimentApproach needs 'sentimentCol' to be set for training")

    val (positive, negative): (Map[String, Long], Map[String, Long]) = {
      import ResourceHelper.spark.implicits._
      val positiveDS = new MapAccumulator()
      val negativeDS = new MapAccumulator()
      dataset.sparkSession.sparkContext.register(positiveDS)
      dataset.sparkSession.sparkContext.register(negativeDS)
      val prefix = "not_"
      val tokenColumn = dataset.schema.fields
        .find(f =>
          f.metadata.contains("annotatorType") && f.metadata
            .getString("annotatorType") == AnnotatorType.TOKEN)
        .map(_.name)
        .get

      dataset
        .select(tokenColumn, $(sentimentCol))
        .as[(Array[Annotation], String)]
        .foreach(tokenSentiment => {
          negateSequence(tokenSentiment._1.map(_.result)).foreach(w => {
            if (tokenSentiment._2 == "positive") {
              positiveDS.add(w, 1)
              negativeDS.add(prefix + w, 1)
            } else if (tokenSentiment._2 == "negative") {
              negativeDS.add(w, 1)
              positiveDS.add(prefix + w, 1)
            }
          })
        })
      (positiveDS.value.withDefaultValue(0), negativeDS.value.withDefaultValue(0))
    }

    val positiveTotals = positive.values.sum
    val negativeTotals = negative.values.sum

    def mutualInformation(word: String): Double = {
      val T = positiveTotals + negativeTotals
      val W = positive(word) + negative(word)
      var I: Double = 0.0
      if (W == 0) {
        return 0
      }
      if (negative(word) > 0) {
        val negativeDeltaScore: Double =
          (negativeTotals - negative(word)) * T / (T - W) / negativeTotals
        I += (negativeTotals - negative(word)) / T * scala.math.log(negativeDeltaScore)
        val negativeScore: Double = negative(word) * T / W / negativeTotals
        I += negative(word) / T * scala.math.log(negativeScore)
      }
      if (positive(word) > 0) {
        val positiveDeltaScore: Double =
          (positiveTotals - positive(word)) * T / (T - W) / positiveTotals
        I += (positiveTotals - positive(word)) / T * scala.math.log(positiveDeltaScore)
        val positiveScore: Double = positive(word) * T / W / positiveTotals
        I += positive(word) / T * scala.math.log(positiveScore)
      }
      I
    }

    val words = (positive.keys ++ negative.keys).toArray.distinct.sortBy(-mutualInformation(_))

    new ViveknSentimentModel()
      .setImportantFeatureRatio($(importantFeatureRatio))
      .setUnimportantFeatureStep($(unimportantFeatureStep))
      .setFeatureLimit($(featureLimit))
      .setPositive(positive)
      .setNegative(negative)
      .setPositiveTotals(positiveTotals)
      .setNegativeTotals(negativeTotals)
      .setWords(words)
  }

}

private object ViveknSentimentApproach extends DefaultParamsReadable[ViveknSentimentApproach]
