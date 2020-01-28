package com.johnsnowlabs.nlp

import org.apache.spark.ml.{Model, PipelineModel}
import org.apache.spark.sql.Dataset

/** AnnotatorApproach'es may extend this trait in order to allow
  * RecursivePipelines to include intermediate
  * steps trained PipelineModel's
  * */
trait HasRecursiveFit[M <: Model[M]] {

  this: AnnotatorApproach[M] =>

    final def recursiveFit(dataset: Dataset[_], recursivePipeline: PipelineModel): M = {
      _fit(dataset, Some(recursivePipeline))
    }

}
