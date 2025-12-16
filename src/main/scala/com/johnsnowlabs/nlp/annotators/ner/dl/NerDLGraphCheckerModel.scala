/*
 * Copyright 2017-2025 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp._
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.{IntParam, LongParam, Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{MetadataBuilder, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

/** Resulting model from [[NerDLGraphChecker]], that updates dataframe metadata (label column)
  * with NerDLGraph parameters. It does not perform any actual data transformations, as the
  * checks/computations are done during the `fit` phase.
  *
  * This annotator should never be used directly.
  */
class NerDLGraphCheckerModel(override val uid: String)
    extends Model[NerDLGraphCheckerModel]
    with HasInputAnnotationCols
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("NerDLGraphChecker"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)

  /** Column with label per each token
    *
    * @group param
    */
  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each token")

  /* @group getParam */
  def getLabelColumn: String = $(labelColumn)

  /* @group setParam */
  def setLabelColumn(value: String): this.type = set(labelColumn, value)

  /** Dimensionality of embeddings
    *
    * @group param
    */
  val embeddingsDim = new IntParam(this, "embeddingsDim", "Dimensionality of embeddings")

  /* @group setParam */
  def setEmbeddingsDim(value: Int): this.type = set(embeddingsDim, value)
  /* @group getParam */
  def getEmbeddingsDim: Int = $(embeddingsDim)

  /** Number of labels in the dataset
    *
    * @group param
    */
  val labels = new StringArrayParam(this, "labels", "Labels in the dataset.")
  /* @group setParam */
  def setLabels(labels: Array[String]): this.type = set(this.labels, labels)
  /* @group getParam */
  def getLabels: Array[String] = $(labels)

  /** Maximum number of characters in the dataset
    *
    * @group param
    */
  val chars = new StringArrayParam(this, "chars", "Set of characters in the dataset.")
  /* @group setParam */
  def setChars(chars: Array[String]): this.type = set(this.chars, chars)
  /* @group getParam */
  def getChars: Array[String] = $(chars)

  /** Number of training examples in the dataset
    *
    * @group param
    */
  val dsLen = new LongParam(this, "dsLen", "Length of the training dataset.")

  /* @group setParam */
  def setDsLen(value: Long): this.type = set(dsLen, value)
  /* @group getParam */
  def getDsLen: Long = $(dsLen)

  /** Folder path that contain external graph files
    *
    * @group param
    */
  val graphFolder =
    new Param[String](this, "graphFolder", "Folder path that contain external graph files")

  /* @group setParam */
  def setGraphFolder(value: String): this.type = set(graphFolder, value)

  override def transformSchema(schema: StructType): StructType = schema

  /** Adds metadata with graph parameters to the label column.
    *
    * @param dataset
    *   input dataset
    * @return
    *   transformed dataset with new output column
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val labelCol = getLabelColumn
    val schema = dataset.schema
    val labelField = schema(labelCol)

    // Construct graph params metadata
    val graphParams = new MetadataBuilder()
      .putLong(NerDLGraphCheckerModel.embeddingsDimKey, getEmbeddingsDim)
      .putStringArray(NerDLGraphCheckerModel.labelsKey, getLabels)
      .putStringArray(NerDLGraphCheckerModel.charsKey, getChars)
      .putLong(NerDLGraphCheckerModel.dsLenKey, getDsLen)
      .build()

    val labelFieldMeta = new MetadataBuilder()
      .withMetadata(labelField.metadata)
      .putMetadata(NerDLGraphCheckerModel.graphParamsMetadataKey, graphParams)
      .build()

    dataset.withMetadata(labelCol, labelFieldMeta)
  }

  override def copy(extra: ParamMap): NerDLGraphCheckerModel = defaultCopy(extra)
}

object NerDLGraphCheckerModel extends ParamsAndFeaturesReadable[NerDLGraphCheckerModel] {
  def graphParamsMetadataKey: String = "NerDLGraphCheckerParams"
  def embeddingsDimKey: String = "embeddingsDim"
  def labelsKey: String = "labels"
  def charsKey: String = "chars"
  def dsLenKey: String = "dsLen"
}
