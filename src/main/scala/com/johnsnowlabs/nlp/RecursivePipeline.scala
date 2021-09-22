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

import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{Identifiable, MLWritable, MLWriter}
import org.apache.spark.ml.{Estimator, Model, Pipeline, PipelineModel, PipelineStage, Transformer}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable.ListBuffer

class RecursivePipeline(override val uid: String, baseStages: Array[PipelineStage]) extends Pipeline {

  def this() = this(Identifiable.randomUID("RECURSIVE_PIPELINE"), Array.empty)

  def this(uid: String) = this(uid, Array.empty)

  def this(pipeline: Pipeline) = this(pipeline.uid, pipeline.getStages)

  this.setStages(baseStages)

  /** Workaround to PipelineModel being private in Spark */
  private def createPipeline(dataset: Dataset[_], transformers: Array[Transformer]): PipelineModel = {
    new Pipeline(uid).setStages(transformers).fit(dataset)
  }

  /** Code Duplication for Spark ML. This is not a good practice, but stages logic is tightly coupled on fit and PipelineModel is private[ml] */
  override def fit(dataset: Dataset[_]): PipelineModel = {
    transformSchema(dataset.schema, logging = true)
    val theStages = $(stages)
    var indexOfLastEstimator = -1
    theStages.view.zipWithIndex.foreach { case (stage, index) =>
      stage match {
        case _: Estimator[_] =>
          indexOfLastEstimator = index
        case _ =>
      }
    }
    var curDataset = dataset
    val transformers = ListBuffer.empty[Transformer]
    theStages.view.zipWithIndex.foreach { case (stage, index) =>
      if (index <= indexOfLastEstimator) {
        val transformer = stage match {
          case estimator: HasRecursiveFit[_] =>
            estimator.recursiveFit(curDataset, new Pipeline(uid).setStages(transformers.toArray).fit(dataset))
          case estimator: Estimator[_] =>
            estimator.fit(curDataset)
          case t: Transformer =>
            t
          case _ =>
            throw new IllegalArgumentException(
              s"Does not support stage $stage of type ${stage.getClass}")
        }
        if (index < indexOfLastEstimator) {
          curDataset = transformer.transform(curDataset)
        }
        transformers += transformer
      } else {
        transformers += stage.asInstanceOf[Transformer]
      }
    }

    createPipeline(dataset, transformers.toArray)
  }

}

class RecursivePipelineModel(override val uid: String, innerPipeline: PipelineModel)
  extends Model[RecursivePipelineModel] with MLWritable with Logging {

  def this(pipeline: PipelineModel) = this(pipeline.uid, pipeline)

  // drops right at most because is itself included
  private def createRecursiveAnnotators(dataset: Dataset[_]): PipelineModel =
    new Pipeline(uid).setStages(innerPipeline.stages.dropRight(1)).fit(dataset)

  override def copy(extra: ParamMap): RecursivePipelineModel = {
    new RecursivePipelineModel(uid, innerPipeline.copy(extra))
  }

  override def write: MLWriter = {
    innerPipeline.write
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    innerPipeline.stages.foldLeft(dataset.toDF)((cur, transformer) => transformer match {
      case t: HasRecursiveTransform[_] => t.recursiveTransform(cur, createRecursiveAnnotators(dataset))
      case t: AnnotatorModel[_] if t.getLazyAnnotator => cur
      case t: Transformer => t.transform(cur)
    })
  }

  override def transformSchema(schema: StructType): StructType = {
    innerPipeline.transformSchema(schema)
  }
}
