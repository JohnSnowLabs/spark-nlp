#  Copyright 2017-2021 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import unittest
from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest

"""
----
CREATE THE FOLLOWING SCALA CLASS IN ORDER TO RUN THIS TEST
----

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.TokenizerModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.Dataset

class SomeApproachTest(override val uid: String) extends AnnotatorApproach[SomeModelTest] with HasRecursiveFit[SomeModelTest] {
  override val description: String = "Some Approach"

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): SomeModelTest = {
    require(recursivePipeline.isDefined)
    require(recursivePipeline.get.stages.length == 2)
    require(recursivePipeline.get.stages.last.isInstanceOf[TokenizerModel])
    new SomeModelTest()
  }

  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val outputAnnotatorType: AnnotatorType = "BAR"
}

class SomeModelTest(override val uid: String) extends AnnotatorModel[SomeModelTest] with HasRecursiveTransform[SomeModelTest] {

  def this() = this("bar_uid")

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    require(recursivePipeline.isDefined)
    require(recursivePipeline.get.stages.length == 2)
    require(recursivePipeline.get.stages.last.isInstanceOf[TokenizerModel])
    Seq.empty
  }

  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val outputAnnotatorType: AnnotatorType = "BAR"
}
"""


class SomeAnnotatorTest(AnnotatorApproach, HasRecursiveFit):

    def __init__(self):
        super(SomeAnnotatorTest, self).__init__(classname="com.johnsnowlabs.nlp.SomeApproachTest")

    def _create_model(self, java_model):
        return SomeModelTest(java_model=java_model)


class SomeModelTest(AnnotatorModel, HasRecursiveTransform):

    def __init__(self, classname="com.johnsnowlabs.nlp.SomeModelTest", java_model=None):
        super(SomeModelTest, self).__init__(
            classname=classname,
            java_model=java_model
        )


class RecursiveTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        some_annotator = SomeAnnotatorTest() \
            .setInputCols(['token']) \
            .setOutputCol('baaar')
        pipeline = RecursivePipeline().setStages([document_assembler, tokenizer, some_annotator])
        model = pipeline.fit(self.data)
        RecursivePipelineModel(model).transform(self.data).show()
