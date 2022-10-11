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

package com.johnsnowlabs.nlp.annotators.ws

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{
  PerceptronTrainingUtils,
  TrainingPerceptronLegacy
}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.{Map => MMap}

/** Trains a WordSegmenter which tokenizes non-english or non-whitespace separated texts.
  *
  * Many languages are not whitespace separated and their sentences are a concatenation of many
  * symbols, like Korean, Japanese or Chinese. Without understanding the language, splitting the
  * words into their corresponding tokens is impossible. The WordSegmenter is trained to
  * understand these languages and split them into semantically correct parts.
  *
  * For instantiated/pretrained models, see [[WordSegmenterModel]].
  *
  * To train your own model, a training dataset consisting of
  * [[https://en.wikipedia.org/wiki/Part-of-speech_tagging Part-Of-Speech tags]] is required. The
  * data has to be loaded into a dataframe, where the column is an
  * [[com.johnsnowlabs.nlp.Annotation Annotation]] of type `"POS"`. This can be set with
  * `setPosColumn`.
  *
  * '''Tip''': The helper class [[com.johnsnowlabs.nlp.training.POS POS]] might be useful to read
  * training data into data frames.
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/annotation/chinese/word_segmentation Spark NLP Workshop]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/WordSegmenterTest.scala WordSegmenterTest]].
  *
  * ==Example==
  * In this example, `"chinese_train.utf8"` is in the form of
  * {{{
  * 十|LL 四|RR 不|LL 是|RR 四|LL 十|RR
  * }}}
  * and is loaded with the `POS` class to create a dataframe of `"POS"` type Annotations.
  * {{{
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.ws.WordSegmenterApproach
  * import com.johnsnowlabs.nlp.training.POS
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val wordSegmenter = new WordSegmenterApproach()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *   .setPosColumn("tags")
  *   .setNIterations(5)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   wordSegmenter
  * ))
  *
  * val trainingDataSet = POS().readDataset(
  *   spark,
  *   "src/test/resources/word-segmenter/chinese_train.utf8"
  * )
  *
  * val pipelineModel = pipeline.fit(trainingDataSet)
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
class WordSegmenterApproach(override val uid: String)
    extends AnnotatorApproach[WordSegmenterModel]
    with PerceptronTrainingUtils {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("WORD_SEGMENTER"))

  override val description: String = "Word segmentation"

  /** Column of Array of POS tags that match tokens
    *
    * @group param
    */
  val posCol = new Param[String](this, "posCol", "column of Array of POS tags that match tokens")

  /** Number of iterations in training, converges to better accuracy (Default: `5`)
    *
    * @group param
    */
  val nIterations = new IntParam(
    this,
    "nIterations",
    "Number of iterations in training, converges to better accuracy")

  /** How many times at least a tag on a word to be marked as frequent (Default: `20`)
    *
    * @group param
    */
  val frequencyThreshold = new IntParam(
    this,
    "frequencyThreshold",
    "How many times at least a tag on a word to be marked as frequent")

  /** How much percentage of total amount of words are covered to be marked as frequent (Default:
    * `0.97`)
    *
    * @group param
    */
  val ambiguityThreshold = new DoubleParam(
    this,
    "ambiguityThreshold",
    "How much percentage of total amount of words are covered to be marked as frequent")

  val enableRegexTokenizer: BooleanParam = new BooleanParam(
    this,
    "enableRegexTokenizer",
    "Whether to use RegexTokenizer before segmentation. Useful for multilingual text")

  /** Indicates whether to convert all characters to lowercase before tokenizing (Default:
    * `false`).
    *
    * @group param
    */
  val toLowercase: BooleanParam = new BooleanParam(
    this,
    "toLowercase",
    "Indicates whether to convert all characters to lowercase before tokenizing. Used only when enableRegexTokenizer is true")

  /** Regex pattern used to match delimiters (Default: `"\\s+"`)
    *
    * @group param
    */
  val pattern: Param[String] = new Param(
    this,
    "pattern",
    "regex pattern used for tokenizing. Used only when enableRegexTokenizer is true")

  /** @group setParam */
  def setPosColumn(value: String): this.type = set(posCol, value)

  /** @group setParam */
  def setNIterations(value: Int): this.type = set(nIterations, value)

  /** @group setParam */
  def setFrequencyThreshold(value: Int): this.type = set(frequencyThreshold, value)

  /** @group setParam */
  def setAmbiguityThreshold(value: Double): this.type = set(ambiguityThreshold, value)

  /** @group setParam */
  def setEnableRegexTokenizer(value: Boolean): this.type = set(enableRegexTokenizer, value)

  /** @group setParam */
  def setToLowercase(value: Boolean): this.type = set(toLowercase, value)

  /** @group setParam */
  def setPattern(value: String): this.type = set(pattern, value)

  setDefault(
    nIterations -> 5,
    frequencyThreshold -> 20,
    ambiguityThreshold -> 0.97,
    enableRegexTokenizer -> false,
    toLowercase -> false,
    pattern -> "\\s+")

  /** @group getParam */
  def getNIterations: Int = $(nIterations)

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): WordSegmenterModel = {
    val taggedSentences = generatesTagBook(dataset)
    val taggedWordBook =
      buildTagBook(taggedSentences, $(frequencyThreshold), $(ambiguityThreshold))

    /** Finds all distinct tags and stores them */
    val classes = taggedSentences.flatMap(_.tags).distinct
    val initialModel = new TrainingPerceptronLegacy(classes, taggedWordBook, MMap())
    val finalModel =
      trainPerceptron($(nIterations), initialModel, taggedSentences, taggedWordBook)

    new WordSegmenterModel()
      .setModel(finalModel)
      .setEnableRegexTokenizer($(enableRegexTokenizer))
      .setToLowercase($(toLowercase))
      .setPattern($(pattern))
  }

  /** Output Annotator Types: TOKEN
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** Input Annotator Types: DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)
}

/** This is the companion object of [[WordSegmenterApproach]]. Please refer to that class for the
  * documentation.
  */
object WordSegmenterApproach extends DefaultParamsReadable[WordSegmenterApproach]
