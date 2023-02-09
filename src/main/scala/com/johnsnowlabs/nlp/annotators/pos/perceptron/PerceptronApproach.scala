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

package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.AnnotatorApproach
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.{Map => MMap}

/** Trains an averaged Perceptron model to tag words part-of-speech. Sets a POS tag to each word
  * within a sentence.
  *
  * For pretrained models please see the [[PerceptronModel]].
  *
  * The training data needs to be in a Spark DataFrame, where the column needs to consist of
  * [[com.johnsnowlabs.nlp.Annotation Annotations]] of type `POS`. The `Annotation` needs to have
  * member `result` set to the POS tag and have a `"word"` mapping to its word inside of member
  * `metadata`. This DataFrame for training can easily created by the helper class
  * [[com.johnsnowlabs.nlp.training.POS POS]].
  * {{{
  * POS().readDataset(spark, datasetPath).selectExpr("explode(tags) as tags").show(false)
  * +---------------------------------------------+
  * |tags                                         |
  * +---------------------------------------------+
  * |[pos, 0, 5, NNP, [word -> Pierre], []]       |
  * |[pos, 7, 12, NNP, [word -> Vinken], []]      |
  * |[pos, 14, 14, ,, [word -> ,], []]            |
  * |[pos, 31, 34, MD, [word -> will], []]        |
  * |[pos, 36, 39, VB, [word -> join], []]        |
  * |[pos, 41, 43, DT, [word -> the], []]         |
  * |[pos, 45, 49, NN, [word -> board], []]       |
  *                       ...
  * }}}
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/french/Train-Perceptron-French.ipynb Examples]]
  * and
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/pos/perceptron PerceptronApproach tests]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.SentenceDetector
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.training.POS
  * import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentence = new SentenceDetector()
  *   .setInputCols("document")
  *   .setOutputCol("sentence")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("sentence")
  *   .setOutputCol("token")
  *
  * val datasetPath = "src/test/resources/anc-pos-corpus-small/test-training.txt"
  * val trainingPerceptronDF = POS().readDataset(spark, datasetPath)
  *
  * val trainedPos = new PerceptronApproach()
  *   .setInputCols("document", "token")
  *   .setOutputCol("pos")
  *   .setPosColumn("tags")
  *   .fit(trainingPerceptronDF)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   sentence,
  *   tokenizer,
  *   trainedPos
  * ))
  *
  * val data = Seq("To be or not to be, is this the question?").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("pos.result").show(false)
  * +--------------------------------------------------+
  * |result                                            |
  * +--------------------------------------------------+
  * |[NNP, NNP, CD, JJ, NNP, NNP, ,, MD, VB, DT, CD, .]|
  * +--------------------------------------------------+
  * }}}
  *
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
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class PerceptronApproach(override val uid: String)
    extends AnnotatorApproach[PerceptronModel]
    with PerceptronTrainingUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Averaged Perceptron model to tag words part-of-speech */
  override val description: String = "Averaged Perceptron model to tag words part-of-speech"

  /** Column of Array of POS tags that match tokens
    *
    * @group param
    */
  val posCol = new Param[String](this, "posCol", "Column of Array of POS tags that match tokens")

  /** Number of iterations in training, converges to better accuracy (Default: `5`)
    *
    * @group param
    */
  val nIterations = new IntParam(
    this,
    "nIterations",
    "Number of iterations in training, converges to better accuracy")
  setDefault(nIterations, 5)

  /** How many times at least a tag on a word to be marked as frequent (Default: `20`)
    *
    * @group param
    */
  val frequencyThreshold = new IntParam(
    this,
    "frequencyThreshold",
    "How many times at least a tag on a word to be marked as frequent")
  setDefault(frequencyThreshold, 20)

  /** How much percentage of total amount of words are covered to be marked as frequent (Default:
    * `0.97`)
    *
    * @group param
    */
  val ambiguityThreshold = new DoubleParam(
    this,
    "ambiguityThreshold",
    "How much percentage of total amount of words are covered to be marked as frequent")
  setDefault(ambiguityThreshold, 0.97)

  /** Column containing an array of POS Tags matching every token on the line.
    *
    * @group setParam
    */
  def setPosColumn(value: String): this.type = set(posCol, value)

  /** Number of iterations for training. May improve accuracy but takes longer. Default 5.
    *
    * @group setParam
    */
  def setNIterations(value: Int): this.type = set(nIterations, value)

  /** "How many times at least a tag on a word to be marked as frequent
    *
    * @group setParam
    */
  def setFrequencyThreshold(value: Int): this.type = set(frequencyThreshold, value)

  /** "How much percentage of total amount of words are covered to be marked as frequent
    *
    * @group setParam
    */
  def setAmbiguityThreshold(value: Double): this.type = set(ambiguityThreshold, value)

  /** Number of iterations for training. May improve accuracy but takes longer (Default: `5`)
    *
    * @group getParam
    */
  def getNIterations: Int = $(nIterations)

  def this() = this(Identifiable.randomUID("POS"))

  /** Output annotator type: POS
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = POS

  /** Input annotator type: TOKEN, DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  /** Trains a model based on a provided CORPUS
    *
    * @return
    *   A trained averaged model
    */
  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): PerceptronModel = {

    val taggedSentences = generatesTagBook(dataset)
    val taggedWordBook =
      buildTagBook(taggedSentences, $(frequencyThreshold), $(ambiguityThreshold))

    /** finds all distinct tags and stores them */
    val classes = taggedSentences.flatMap(_.tags).distinct
    val initialModel = new TrainingPerceptronLegacy(classes, taggedWordBook, MMap())
    val finalModel =
      trainPerceptron($(nIterations), initialModel, taggedSentences, taggedWordBook)
    logger.debug("TRAINING: Finished all iterations")

    new PerceptronModel().setModel(finalModel)
  }

}

/** This is the companion object of [[PerceptronApproach]]. Please refer to that class for the
  * documentation.
  */
object PerceptronApproach extends DefaultParamsReadable[PerceptronApproach]
