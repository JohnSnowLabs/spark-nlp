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