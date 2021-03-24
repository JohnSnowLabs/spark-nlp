package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.TokenizerModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.Dataset

class SomeApproachTest(override val uid: String) extends AnnotatorApproach[SomeModelTest] with HasRecursiveFit[SomeModelTest] {
  override val description: String = "Some Approach"

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): SomeModelTest = {
    require(recursivePipeline.isDefined, "RecursiveApproach Did not receive any recursive pipelines")
    require(recursivePipeline.get.stages.length == 2, "RecursiveApproach Did not receive exactly two stages in the recursive pipeline")
    require(recursivePipeline.get.stages.last.isInstanceOf[TokenizerModel], "RecursiveApproach Last stage of recursive pipeline is not the last stage of the recursive pipeline")
    new SomeModelTest()
  }

  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val outputAnnotatorType: AnnotatorType = "BAR"
}

class SomeModelTest(override val uid: String) extends AnnotatorModel[SomeModelTest] with HasSimpleAnnotate[SomeModelTest] with HasRecursiveTransform[SomeModelTest] {

  def this() = this("bar_uid")

  override def annotate(annotations: Seq[Annotation], recursivePipeline: PipelineModel): Seq[Annotation] = {
    require(recursivePipeline.stages.length == 2, "RecursiveModel Did not receive exactly two stages in the recursive pipeline")
    require(recursivePipeline.stages.last.isInstanceOf[TokenizerModel], "RecursiveModel Last stage of recursive pipeline is not the last stage of the recursive pipeline")
    Seq.empty
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    throw new IllegalStateException("SomeModelTest does not have an annotate that works without recursion")
  }

  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val outputAnnotatorType: AnnotatorType = "BAR"
}