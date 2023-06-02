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

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorType.{TOKEN, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp.{AnnotatorApproach, HasEnableCachingProperties, HasProtectedParams}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{DoubleParam, IntParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.sql.{Dataset, SparkSession}

/** Trains a Word2Vec model that creates vector representations of words in a text corpus.
  *
  * The algorithm first constructs a vocabulary from the corpus and then learns vector
  * representation of words in the vocabulary. The vector representation can be used as features
  * in natural language processing and machine learning algorithms.
  *
  * We use Word2Vec implemented in Spark ML. It uses skip-gram model in our implementation and a
  * hierarchical softmax method to train the model. The variable names in the implementation match
  * the original C implementation.
  *
  * For instantiated/pretrained models, see [[Word2VecModel]].
  *
  * '''Sources''' :
  *
  * For the original C implementation, see https://code.google.com/p/word2vec/
  *
  * For the research paper, see
  * [[https://arxiv.org/abs/1301.3781 Efficient Estimation of Word Representations in Vector Space]]
  * and
  * [[https://arxiv.org/pdf/1310.4546v1.pdf Distributed Representations of Words and Phrases and their Compositionality]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.annotator.{Tokenizer, Word2VecApproach}
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols(Array("document"))
  *   .setOutputCol("token")
  *
  * val embeddings = new Word2VecApproach()
  *   .setInputCols("token")
  *   .setOutputCol("embeddings")
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(
  *     documentAssembler,
  *     tokenizer,
  *     embeddings
  *   ))
  *
  * val path = "src/test/resources/spell/sherlockholmes.txt"
  * val dataset = spark.sparkContext.textFile(path)
  *   .toDF("text")
  * val pipelineModel = pipeline.fit(dataset)
  * }}}
  *
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
class Word2VecApproach(override val uid: String)
    extends AnnotatorApproach[Word2VecModel]
    with HasStorageRef
    with HasEnableCachingProperties
    with HasProtectedParams {

  def this() = this(Identifiable.randomUID("Word2VecApproach"))

  override val description =
    "Distributed Representations of Words and Phrases and their Compositionality"

  /** Input Annotator Types: TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  /** Output Annotator Types: WORD_EMBEDDINGS
    *
    * @group anno
    */
  override val outputAnnotatorType: String = WORD_EMBEDDINGS

  /** The dimension of the code that you want to transform from words (Default: `100`).
    *
    * @group param
    */
  val vectorSize =
    new IntParam(this, "vectorSize", "the dimension of codes after transforming from words (> 0)")
      .setProtected()

  /** @group setParam */
  def setVectorSize(value: Int): this.type = {
    require(value > 0, s"vector size must be positive but got $value")
    set(vectorSize, value)
    this
  }

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getVectorSize: Int = $(vectorSize)

  /** The window size (context words from [-window, window]) (Default: `5`)
    *
    * @group param
    */
  val windowSize = new IntParam(
    this,
    "windowSize",
    "the window size (context words from [-window, window]) (> 0)")

  /** @group setParam */
  def setWindowSize(value: Int): this.type = {
    require(value > 0, s"Window of words must be positive but got $value")
    set(windowSize, value)
    this
  }

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getWindowSize: Int = $(windowSize)

  /** Number of partitions for sentences of words (Default: `1`).
    *
    * @group param
    */
  val numPartitions =
    new IntParam(this, "numPartitions", "number of partitions for sentences of words (> 0)")

  /** @group setParam */
  def setNumPartitions(value: Int): this.type = {
    require(value > 0, s"Number of partitions must be positive but got $value")
    set(numPartitions, value)
    this
  }

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getNumPartitions: Int = $(numPartitions)

  /** The minimum number of times a token must appear to be included in the word2vec model's
    * vocabulary (Default: `5`).
    *
    * @group param
    */
  val minCount = new IntParam(
    this,
    "minCount",
    "the minimum number of times a token must " +
      "appear to be included in the word2vec model's vocabulary (>= 0)")

  /** @group setParam */
  def setMinCount(value: Int): this.type = {
    require(value > 0, s"Minimum number of times must be nonnegative but got $value")
    set(minCount, value)
    this
  }

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getMinCount: Int = $(minCount)

  /** Sets the maximum length (in words) of each sentence in the input data (Default: `1000`). Any
    * sentence longer than this threshold will be divided into chunks of up to `maxSentenceLength`
    * size.
    *
    * @group param
    */
  val maxSentenceLength = new IntParam(
    this,
    "maxSentenceLength",
    "Maximum length " +
      "(in words) of each sentence in the input data. Any sentence longer than this threshold will " +
      "be divided into chunks up to the size (> 0)")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(value > 0, s"Maximum length of sentences must be positive but got $value")
    set(maxSentenceLength, value)
    this
  }

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** Param for Step size to be used for each iteration of optimization (&gt; 0) (Default:
    * `0.025`).
    *
    * @group param
    */
  val stepSize: DoubleParam = new DoubleParam(
    this,
    "stepSize",
    "Step size (learning rate) to be used for each iteration of optimization (> 0)")

  /** @group setParam */
  def setStepSize(value: Double): this.type = {
    require(value > 0, s"Initial step size must be positive but got $value")
    set(stepSize, value)
    this
  }

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getStepSize: Double = $(stepSize)

  /** Param for maximum number of iterations (&gt;= 0) (Default: `1`)
    *
    * @group param
    */
  val maxIter: IntParam = new IntParam(this, "maxIter", "maximum number of iterations (>= 0)")

  /** @group setParam */
  def setMaxIter(value: Int): this.type = {
    require(value > 0, s"Number of iterations must be positive but got $value")
    set(maxIter, value)
    this
  }

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getMaxIter: Int = $(maxIter)

  /** Random seed for shuffling the dataset (Default: `44`)
    *
    * @group param
    */
  val seed = new IntParam(this, "seed", "Random seed")

  /** @group setParam */
  def setSeed(value: Int): Word2VecApproach.this.type = {
    require(value > 0, s"random seed must be positive but got $value")
    set(seed, value)
    this
  }

  /** WARNING: this is for internal use and not intended for users
   * @group getParam */
  def getSeed: Int = $(seed)

  setDefault(
    vectorSize -> 100,
    windowSize -> 5,
    numPartitions -> 1,
    minCount -> 1,
    maxSentenceLength -> 1000,
    stepSize -> 0.025,
    maxIter -> 1,
    seed -> 44)

  override def beforeTraining(spark: SparkSession): Unit = {}

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): Word2VecModel = {

    val tokenResult: String = ".result"
    val inputColumns = getInputCols(0) + tokenResult

    val word2Vec = new Word2Vec()
      .setLearningRate($(stepSize))
      .setMinCount($(minCount))
      .setNumIterations($(maxIter))
      .setNumPartitions($(numPartitions))
      .setVectorSize($(vectorSize))
      .setWindowSize($(windowSize))
      .setMaxSentenceLength($(maxSentenceLength))
      .setSeed($(seed))

    val input = dataset.select(dataset.col(inputColumns)).rdd.map(r => r.getSeq[String](0))

    if (getEnableCaching)
      input.cache()

    val model = word2Vec.fit(input)

    if (getEnableCaching)
      input.unpersist()

    new Word2VecModel()
      .setWordVectors(model.getVectors)
      .setVectorSize($(vectorSize))
      .setStorageRef($(storageRef))
      .setDimension($(vectorSize))

  }

}

/** This is the companion object of [[Word2VecApproach]]. Please refer to that class for the
  * documentation.
  */
object Word2VecApproach extends DefaultParamsReadable[Word2VecApproach]
