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
import org.apache.spark.ml.param.{IntParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/** Resulting model from [[NerDLGraphChecker]], that does not perform any transformations, as the
  * checks are done during the `fit` phase. It acts as the identity.
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

  /* @group getParam */
  def getEmbeddingsDim: Int = $(embeddingsDim)

  /* @group setParam */
  def setEmbeddingsDim(value: Int): this.type = set(embeddingsDim, value)

  /** Folder path that contain external graph files
    *
    * @group param
    */
  val graphFolder =
    new Param[String](this, "graphFolder", "Folder path that contain external graph files")

  /* @group setParam */
  def setGraphFolder(value: String): this.type = set(graphFolder, value)

  override def transformSchema(schema: StructType): StructType = schema

  /** Returns the dataset as a dataframe. This annotator does not perform any transformations on
    * the dataset (checks during fit only).
    *
    * @param dataset
    *   input dataset
    * @return
    *   transformed dataset with new output column
    */
  override def transform(dataset: Dataset[_]): DataFrame = dataset.toDF()

  override def copy(extra: ParamMap): NerDLGraphCheckerModel = defaultCopy(extra)
}

object NerDLGraphCheckerModel extends ParamsAndFeaturesReadable[NerDLGraphCheckerModel]
