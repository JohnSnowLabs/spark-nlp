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

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.tensorflow.ClassifierDatasetEncoder
import com.johnsnowlabs.nlp.annotators.param.EvaluationDLParams
import org.apache.spark.ml.param.{FloatParam, IntArrayParam, IntParam, Param}
import org.apache.spark.sql.{DataFrame, Dataset}

trait ClassifierEncoder extends EvaluationDLParams {

  /** Maximum number of epochs to train (Default: `10`)
    *
    * @group param
    */
  val maxEpochs = new IntParam(this, "maxEpochs", "Maximum number of epochs to train")

  /** Learning Rate (Default: `5e-3f`)
    *
    * @group param
    */
  val lr = new FloatParam(this, "lr", "Learning Rate")

  /** Batch size (Default: `64`)
    *
    * @group param
    */
  val batchSize = new IntParam(this, "batchSize", "Batch size")

  /** Column with label per each document
    *
    * @group param
    */
  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each document")

  /** Random seed for shuffling the dataset
    *
    * @group param
    */
  val randomSeed = new IntParam(this, "randomSeed", "Random seed")

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** Maximum number of epochs to train (Default: `10`)
    *
    * @group setParam
    */
  def setMaxEpochs(epochs: Int): this.type = set(maxEpochs, epochs)

  /** Learning Rate (Default: `5e-3f`)
    *
    * @group setParam
    */
  def setLr(lr: Float): this.type = set(this.lr, lr)

  /** Batch size (Default: `64`)
    *
    * @group setParam
    */
  def setBatchSize(batch: Int): this.type = set(this.batchSize, batch)

  /** Column with label per each document
    *
    * @group setParam
    */
  def setLabelColumn(column: String): this.type = set(labelColumn, column)

  /** Random seed
    *
    * @group setParam
    */
  def setRandomSeed(seed: Int): this.type = set(randomSeed, seed)

  /** Tensorflow config Protobytes passed to the TF session
    *
    * @group setParam
    */
  def setConfigProtoBytes(bytes: Array[Int]): this.type =
    set(this.configProtoBytes, bytes)

  /** Maximum number of epochs to train (Default: `10`)
    *
    * @group getParam
    */
  def getMaxEpochs: Int = $(maxEpochs)

  /** Learning Rate (Default: `5e-3f`)
    *
    * @group getParam
    */
  def getLr: Float = $(this.lr)

  /** Batch size (Default: `64`)
    *
    * @group getParam
    */
  def getBatchSize: Int = $(this.batchSize)

  /** Column with label per each document
    *
    * @group getParam
    */
  def getLabelColumn: String = $(this.labelColumn)

  /** Random seed
    *
    * @group getParam
    */
  def getRandomSeed: Int = $(this.randomSeed)

  /** Tensorflow config Protobytes passed to the TF session
    *
    * @group getParam
    */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  setDefault(maxEpochs -> 10, batchSize -> 64)

  protected def buildDatasetWithLabels(
      dataset: Dataset[_],
      inputCols: String): (DataFrame, Array[String]) = {
    val embeddingsField: String = ".embeddings"
    val inputColumns = inputCols + embeddingsField

    val datasetWithLabels =
      dataset.select(dataset.col($(labelColumn)).cast("string"), dataset.col(inputColumns))
    val labels = datasetWithLabels.select($(labelColumn)).distinct.collect.map(x => x(0).toString)

    require(
      labels.length >= 2 && labels.length <= 100,
      s"The total unique number of classes must be more than 2 and less than 100. Currently is ${labels.length}")

    (datasetWithLabels, labels)
  }

  protected def extractInputs(
      encoder: ClassifierDatasetEncoder,
      dataframe: DataFrame): (Array[Array[Float]], Array[String]) = {

    val embeddingsDim = encoder.calculateEmbeddingsDim(dataframe)
    val myClassName = this.getClass.getName.split("\\.").last
    require(
      embeddingsDim <= 1024,
      s"The $myClassName only accepts embeddings less than 1024 dimensions. Current dimension is $embeddingsDim. Please use embeddings" +
        s" with less than ")

    val dataset = encoder.collectTrainingInstances(dataframe, getLabelColumn)
    val inputEmbeddings = encoder.extractSentenceEmbeddings(dataset)
    val inputLabels = encoder.extractLabels(dataset)

    (inputEmbeddings, inputLabels)
  }

}
