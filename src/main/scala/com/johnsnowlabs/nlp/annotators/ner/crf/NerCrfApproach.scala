/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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

/**
  * Algorithm for training Named Entity Recognition Model
  *
  * This Named Entity recognition annotator allows for a generic model to be trained by utilizing a CRF machine learning algorithm. Its train data (train_ner) is either a labeled or an external CoNLL 2003 IOB based spark dataset with Annotations columns. Also the user has to provide word embeddings annotation column.
  * Optionally the user can provide an entity dictionary file for better accuracy
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ner/crf]] for further reference on this API.
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
    **/
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS)
  /** Input annotator types : NAMED_ENTITY
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = NAMED_ENTITY

  /** L2 regularization coefficient
    *
    * @group param
    **/
  val l2 = new DoubleParam(this, "l2", "L2 regularization coefficient")
  /** c0 params defining decay speed for gradient
    *
    * @group param
    **/
  val c0 = new IntParam(this, "c0", "c0 params defining decay speed for gradient")
  /** If Epoch relative improvement less than eps then training is stopped
    *
    * @group param
    **/
  val lossEps = new DoubleParam(this, "lossEps", "If Epoch relative improvement less than eps then training is stopped")
  /** Features with less weights then this param value will be filtered
    *
    * @group param
    **/
  val minW = new DoubleParam(this, "minW", "Features with less weights then this param value will be filtered")
  /** includeConfidence", "whether or not to calculate prediction confidence by token, includes in metadata
    *
    * @group param
    **/
  val includeConfidence = new BooleanParam(this, "includeConfidence", "whether or not to calculate prediction confidence by token, includes in metadata")
  /** Additional dictionaries to use as a features
    *
    * @group param
    **/
  val externalFeatures = new ExternalResourceParam(this, "externalFeatures", "Additional dictionaries to use as a features")

  /** L2 regularization coefficient
    *
    * @group setParam
    **/
  def setL2(l2: Double): this.type = set(this.l2, l2)

  /** c0 params defining decay speed for gradient
    *
    * @group setParam
    **/
  def setC0(c0: Int): this.type = set(this.c0, c0)

  /** If Epoch relative improvement less than eps then training is stopped
    *
    * @group setParam
    **/
  def setLossEps(eps: Double): this.type = set(this.lossEps, eps)

  /** Features with less weights then this param value will be filtered
    *
    * @group setParam
    **/
  def setMinW(w: Double): this.type = set(this.minW, w)

  /** Whether or not to calculate prediction confidence by token, includes in metadata
    *
    * @group setParam
    **/
  def setIncludeConfidence(c: Boolean): this.type = set(includeConfidence, c)

  /** L2 regularization coefficient
    *
    * @group getParam
    **/
  def getL2: Double = $(l2)

  /** c0 params defining decay speed for gradient
    *
    * @group getParam
    **/
  def getC0: Int = $(c0)

  /** If Epoch relative improvement less than eps then training is stopped
    *
    * @group getParam
    **/
  def getLossEps: Double = $(lossEps)

  /** Features with less weights then this param value will be filtered
    *
    * @group getParam
    **/
  def getMinW: Double = $(minW)

  /** Whether or not to calculate prediction confidence by token, includes in metadata
    *
    * @group getParam
    **/
  def getIncludeConfidence: Boolean = $(includeConfidence)

  /** Additional dictionaries to use as a features
    *
    * @group setParam
    **/
  def setExternalFeatures(value: ExternalResource): this.type = {
    require(value.options.contains("delimiter"), "external features is a delimited text. needs 'delimiter' in options")
    set(externalFeatures, value)
  }

  /** Additional dictionaries to use as a features
    *
    * @group setParam
    **/
  def setExternalFeatures(path: String,
                          delimiter: String,
                          readAs: ReadAs.Format = ReadAs.TEXT,
                          options: Map[String, String] = Map("format" -> "text")): this.type =
    set(externalFeatures, ExternalResource(path, readAs, options ++ Map("delimiter" -> delimiter)))

  setDefault(
    minEpochs -> 0,
    maxEpochs -> 1000,
    l2 -> 1f,
    c0 -> 2250000,
    lossEps -> 1e-3f,
    verbose -> Verbose.Silent.id,
    includeConfidence -> false
  )


  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NerCrfModel = {

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
      randomSeed = get(randomSeed)
    )

    val embeddingsRef = HasStorageRef.getStorageRefFromInput(dataset, $(inputCols), AnnotatorType.WORD_EMBEDDINGS)

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

object NerCrfApproach extends DefaultParamsReadable[NerCrfApproach]