package com.johnsnowlabs.nlp

import org.apache.spark.ml.{Model, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset}

trait HasRecursiveTransform[M <: Model[M]] {

  this: AnnotatorModel[M] =>

  def recursiveTransform(dataset: Dataset[_], recursivePipeline: PipelineModel): DataFrame = {
    _transform(dataset, Some(recursivePipeline))
  }

}
