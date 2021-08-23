/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.storage.HasStorage
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.{Estimator, Model, PipelineModel}
import org.apache.spark.sql.types.{ArrayType, MetadataBuilder, StructField, StructType}
import org.apache.spark.sql.{Dataset, SparkSession}

/** This class should grow once we start training on datasets and share params
 * For now it stands as a dummy placeholder for future reference
 */
abstract class AnnotatorApproach[M <: Model[M]]
  extends Estimator[M]
    with HasInputAnnotationCols
    with HasOutputAnnotationCol
    with HasOutputAnnotatorType
    with DefaultParamsWritable
    with CanBeLazy {

  val description: String

  def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel] = None): M

  def beforeTraining(spark: SparkSession): Unit = {}

  def onTrained(model: M, spark: SparkSession): Unit = {}

  /**
   * takes a [[Dataset]] and checks to see if all the required annotation types are present.
   *
   * @param schema to be validated
   * @return True if all the required types are present, else false
   */
  protected def validate(schema: StructType): Boolean = {
    inputAnnotatorTypes.forall {
      inputAnnotatorType =>
        checkSchema(schema, inputAnnotatorType)
    }
  }

  private def indexIfStorage(dataset: Dataset[_]): Unit = {
    this match {
      case withStorage: HasStorage => withStorage.indexStorage(dataset, withStorage.getStoragePath)
      case _ =>
    }
  }

  protected def _fit(dataset: Dataset[_], recursiveStages: Option[PipelineModel]): M = {
    beforeTraining(dataset.sparkSession)
    indexIfStorage(dataset)
    val model = copyValues(train(dataset, recursiveStages).setParent(this))
    onTrained(model, dataset.sparkSession)
    model
  }

  override final def fit(dataset: Dataset[_]): M = {
    _fit(dataset, None)
  }

  override final def copy(extra: ParamMap): Estimator[M] = defaultCopy(extra)

  /** requirement for pipeline transformation validation. It is called on fit() */
  override final def transformSchema(schema: StructType): StructType = {
    require(validate(schema), s"Wrong or missing inputCols annotators in $uid.\n" +
      msgHelper(schema) +
      s"\nMake sure such annotators exist in your pipeline, " +
      s"with the right output names and that they have following annotator types: " +
      s"${inputAnnotatorTypes.mkString(", ")}")
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    val outputFields = schema.fields :+
      StructField(getOutputCol, ArrayType(Annotation.dataType), nullable = false, metadataBuilder.build)
    StructType(outputFields)
  }
}
