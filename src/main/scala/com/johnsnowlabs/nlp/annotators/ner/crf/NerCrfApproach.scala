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

package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.ml.crf.{CrfParams, LinearChainCrf}
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.NerTagged
import com.johnsnowlabs.nlp.annotators.ner.{NerApproach, Verbose}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/** Algorithm for training a Named Entity Recognition Model
  *
  * For instantiated/pretrained models, see [[NerCrfModel]].
  *
  * This Named Entity recognition annotator allows for a generic model to be trained by utilizing
  * a CRF machine learning algorithm. The training data should be a labeled Spark Dataset, e.g.
  * [[com.johnsnowlabs.nlp.training.CoNLL CoNLL]] 2003 IOB with `Annotation` type columns. The
  * data should have columns of type `DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS` and an additional
  * label column of annotator type `NAMED_ENTITY`. Excluding the label, this can be done with for
  * example
  *   - a [[com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector SentenceDetector]],
  *   - a [[com.johnsnowlabs.nlp.annotators.Tokenizer Tokenizer]],
  *   - a [[com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel PerceptronModel]] and
  *   - a [[com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel WordEmbeddingsModel]].
  *
  * Optionally the user can provide an entity dictionary file with [[setExternalFeatures]] for
  * better accuracy.
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/crf-ner/ner_dl_crf.ipynb Spark NLP Workshop]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ner/crf/NerCrfApproachTestSpec.scala NerCrfApproachTestSpec]].
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
  * import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
  * import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
  * import com.johnsnowlabs.nlp.training.CoNLL
  * import com.johnsnowlabs.nlp.annotator.NerCrfApproach
  * import org.apache.spark.ml.Pipeline
  *
  * // This CoNLL dataset already includes a sentence, token, POS tags and label
  * // column with their respective annotator types. If a custom dataset is used,
  * // these need to be defined with for example:
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
  * val posTagger = PerceptronModel.pretrained()
  *   .setInputCols("sentence", "token")
  *   .setOutputCol("pos")
  *
  * // Then the training can start
  * val embeddings = WordEmbeddingsModel.pretrained()
  *   .setInputCols("sentence", "token")
  *   .setOutputCol("embeddings")
  *   .setCaseSensitive(false)
  *
  * val nerTagger = new NerCrfApproach()
  *   .setInputCols("sentence", "token", "pos", "embeddings")
  *   .setLabelColumn("label")
  *   .setMinEpochs(1)
  *   .setMaxEpochs(3)
  *   .setOutputCol("ner")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   embeddings,
  *   nerTagger
  * ))
  *
  * // We use the sentences, tokens, POS tags and labels from the CoNLL dataset.
  * val conll = CoNLL()
  * val trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")
  *
  * val pipelineModel = pipeline.fit(trainingData)
  * }}}
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach NerDLApproach]] for a deep learning
  *   based approach
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.ner.NerConverter NerConverter]] to further process the
  *   results
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
class NerCrfApproach(override val uid: String)
    extends AnnotatorApproach[NerCrfModel]
    with NerApproach[NerCrfApproach] {

  def this() = this(Identifiable.randomUID("NER"))

  /** CRF based Named Entity Recognition Tagger */
  override val description = "CRF based Named Entity Recognition Tagger"

  /** Input annotator types : DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS)

  /** Output annotator types : NAMED_ENTITY
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = NAMED_ENTITY

  /** L2 regularization coefficient (Default: `1f`)
    *
    * @group param
    */
  val l2 = new DoubleParam(this, "l2", "L2 regularization coefficient")

  /** c0 params defining decay speed for gradient (Default: `2250000`)
    *
    * @group param
    */
  val c0 = new IntParam(this, "c0", "c0 params defining decay speed for gradient")

  /** If Epoch relative improvement is less than `lossEps` then training is stopped (Default:
    * `1e-3f`)
    *
    * @group param
    */
  val lossEps = new DoubleParam(
    this,
    "lossEps",
    "If Epoch relative improvement less than eps then training is stopped")

  /** Features with less weights then this param value will be filtered
    *
    * @group param
    */
  val minW = new DoubleParam(
    this,
    "minW",
    "Features with less weights then this param value will be filtered")

  /** Whether or not to calculate prediction confidence by token, included in metadata (Default:
    * `false`)
    *
    * @group param
    */
  val includeConfidence = new BooleanParam(
    this,
    "includeConfidence",
    "whether or not to calculate prediction confidence by token, includes in metadata")

  /** Additional dictionary to use for features
    *
    * @group param
    */
  val externalFeatures = new ExternalResourceParam(
    this,
    "externalFeatures",
    "Additional dictionary to use for features")

  /** Level of verbosity during training (Default: `Verbose.Silent.id`)
    *
    * @group param
    */
  val verbose = new IntParam(this, "verbose", "Level of verbosity during training")

  /** L2 regularization coefficient
    *
    * @group setParam
    */
  def setL2(l2: Double): this.type = set(this.l2, l2)

  /** c0 params defining decay speed for gradient
    *
    * @group setParam
    */
  def setC0(c0: Int): this.type = set(this.c0, c0)

  /** If Epoch relative improvement less than eps then training is stopped
    *
    * @group setParam
    */
  def setLossEps(eps: Double): this.type = set(this.lossEps, eps)

  /** Features with less weights then this param value will be filtered
    *
    * @group setParam
    */
  def setMinW(w: Double): this.type = set(this.minW, w)

  /** Whether or not to calculate prediction confidence by token, includes in metadata
    *
    * @group setParam
    */
  def setIncludeConfidence(c: Boolean): this.type = set(includeConfidence, c)

  /** Level of verbosity during training (Default: `Verbose.Silent.id`)
    *
    * @group setParam
    */
  def setVerbose(verbose: Int): this.type = set(this.verbose, verbose)

  /** Level of verbosity during training (Default: `Verbose.Silent.id`)
    *
    * @group setParam
    */
  def setVerbose(verbose: Verbose.Level): this.type =
    set(this.verbose, verbose.id)

  /** L2 regularization coefficient
    *
    * @group getParam
    */
  def getL2: Double = $(l2)

  /** c0 params defining decay speed for gradient
    *
    * @group getParam
    */
  def getC0: Int = $(c0)

  /** If Epoch relative improvement less than eps then training is stopped
    *
    * @group getParam
    */
  def getLossEps: Double = $(lossEps)

  /** Features with less weights then this param value will be filtered
    *
    * @group getParam
    */
  def getMinW: Double = $(minW)

  /** Whether or not to calculate prediction confidence by token, includes in metadata
    *
    * @group getParam
    */
  def getIncludeConfidence: Boolean = $(includeConfidence)

  /** Additional dictionary to use for features
    *
    * @group setParam
    */
  def setExternalFeatures(value: ExternalResource): this.type = {
    require(
      value.options.contains("delimiter"),
      "external features is a delimited text. needs 'delimiter' in options")
    set(externalFeatures, value)
  }

  /** Additional dictionary to use for features
    *
    * @group setParam
    */
  def setExternalFeatures(
      path: String,
      delimiter: String,
      readAs: ReadAs.Format = ReadAs.TEXT,
      options: Map[String, String] = Map("format" -> "text")): this.type =
    set(
      externalFeatures,
      ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))

  setDefault(
    minEpochs -> 0,
    maxEpochs -> 1000,
    l2 -> 1f,
    c0 -> 2250000,
    lossEps -> 1e-3f,
    verbose -> Verbose.Silent.id,
    includeConfidence -> false)

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): NerCrfModel = {

    val rows = dataset.toDF()

    val trainDataset =
      NerTagged.collectTrainingInstancesWithPos(rows, getInputCols, $(labelColumn))

    val extraFeatures = get(externalFeatures)
    val dictFeatures = DictionaryFeatures.read(extraFeatures)
    val crfDataset = FeatureGenerator(dictFeatures)
      .generateDataset(trainDataset)

    val params = CrfParams(
      minEpochs = getOrDefault(minEpochs),
      maxEpochs = getOrDefault(maxEpochs),
      l2 = getOrDefault(l2).toFloat,
      c0 = getOrDefault(c0),
      lossEps = getOrDefault(lossEps).toFloat,
      verbose = Verbose.Epochs,
      randomSeed = get(randomSeed))

    val embeddingsRef =
      HasStorageRef.getStorageRefFromInput(dataset, $(inputCols), AnnotatorType.WORD_EMBEDDINGS)

    val crf = new LinearChainCrf(params)
    val crfModel = crf.trainSGD(crfDataset)

    var model = new NerCrfModel()
      .setModel(crfModel)
      .setDictionaryFeatures(dictFeatures)
      .setIncludeConfidence($(includeConfidence))
      .setStorageRef(embeddingsRef)

    if (isDefined(entities))
      model.setEntities($(entities))

    if (isDefined(minW))
      model = model.shrink($(minW).toFloat)

    model
  }
}

/** This is the companion object of [[NerCrfApproach]]. Please refer to that class for the
  * documentation.
  */
object NerCrfApproach extends DefaultParamsReadable[NerCrfApproach]
