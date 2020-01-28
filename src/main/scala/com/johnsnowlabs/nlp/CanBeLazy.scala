package com.johnsnowlabs.nlp

import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.param.BooleanParam

trait CanBeLazy {
  this: PipelineStage =>

  val lazyAnnotator: BooleanParam = new BooleanParam(this, "lazyAnnotator", "Whether this AnnotatorModel acts as lazy in RecursivePipelines")
  def setLazyAnnotator(value: Boolean): this.type = set(lazyAnnotator, value)
  def getLazyAnnotator: Boolean = $(lazyAnnotator)
  setDefault(lazyAnnotator, false)

}
