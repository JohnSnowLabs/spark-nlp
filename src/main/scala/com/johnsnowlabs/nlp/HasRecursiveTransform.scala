package com.johnsnowlabs.nlp

import org.apache.spark.ml.{Model, PipelineModel}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, Dataset}

trait HasRecursiveTransform[M <: Model[M]] {

  this: AnnotatorModel[M] =>

  def annotate(annotations: Seq[Annotation], recursivePipeline: PipelineModel): Seq[Annotation]

  def dfRecAnnotate(recursivePipeline: PipelineModel): UserDefinedFunction = udf {
    annotationProperties: Seq[AnnotationContent] =>
      annotate(annotationProperties.flatMap(_.map(Annotation(_))), recursivePipeline)
  }

  def recursiveTransform(dataset: Dataset[_], recursivePipeline: PipelineModel): DataFrame = {
    _transform(dataset, Some(recursivePipeline))
  }

}
