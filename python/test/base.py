#  Copyright 2017-2022 John Snow Labs
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
from test.common import UpperCaseSparkNLPTransformer
from test.util import SparkContextForTest, SparkSessionForTest

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


class CustomAnnotatorLightPipelineTest(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSessionForTest.spark

    def runTest(self):
        text = "In London, John Snow is a Physician. In Castle Black, Jon Snow is a Lord Commander"
        data = self.spark.createDataFrame([[text]]).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        upper_case = UpperCaseSparkNLPTransformer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("upper")

        pipeline = Pipeline(stages=[document_assembler, sentence_detector, upper_case])
        pipeline_model = pipeline.fit(data)
        # pipeline_model.transform(data).show(truncate=False)
        light_pipeline = LightPipeline(pipeline_model)
        result_annotate = light_pipeline.annotate(text)
        print(result_annotate)
        result_full_annotate = light_pipeline.fullAnnotate(text)
        print(result_full_annotate)


class OnlyAnnotatorsLightPipelineTest(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSessionForTest.spark

    def runTest(self):
        text = "In London, John Snow is a Physician. In Castle Black, Jon Snow is a Lord Commander"
        data = self.spark.createDataFrame([[text]]).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")

        pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer])
        pipeline_model = pipeline.fit(data)
        light_pipeline = LightPipeline(pipeline_model)
        result_annotate = light_pipeline.annotate(text)
        print(result_annotate)
        result_full_annotate = light_pipeline.fullAnnotate(text)
        print(result_full_annotate)


class ThreeCustomAnnotatorsTwoAnnotatorsLightPipelineTest(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSessionForTest.spark

    def runTest(self):

        data = self.spark.createDataFrame([["A simple example"]]).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        stages = [document_assembler]

        input_col = "document"
        for i in range(1, 4):
            output_col = "upper_" + str(i)
            stages.append(UpperCaseSparkNLPTransformer().setInputCols([input_col]).setOutputCol(output_col))
            input_col = output_col

        for i in range(1, 3):
            output_col = "sentence_" + str(i)
            stages.append(SentenceDetector().setInputCols([input_col]).setOutputCol(output_col))
            input_col = output_col

        pipeline = Pipeline(stages=stages)
        # pipeline.fit(data).transform(data).show(truncate=False)
        pipeline_model = pipeline.fit(data)
        light_pipeline = LightPipeline(pipeline_model)
        result = light_pipeline.annotate("light example")
        print(result)


class CustomAnnotatorLongLightPipelineTest(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSessionForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([["A simple example"]]).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector_1 = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence_1")

        upper_case_1 = UpperCaseSparkNLPTransformer() \
            .setInputCols(["sentence_1"]) \
            .setOutputCol("upper_1")

        sentence_detector_2 = SentenceDetector() \
            .setInputCols(["upper_1"]) \
            .setOutputCol("sentence_2")

        stages = [document_assembler, sentence_detector_1, upper_case_1, sentence_detector_2]

        input_col = "upper_1"
        for i in range(2, 5):
            output_col = "upper_" + str(i)
            stages.append(UpperCaseSparkNLPTransformer().setInputCols([input_col]).setOutputCol(output_col))
            input_col = output_col

        for i in range(3, 5):
            output_col = "sentence_" + str(i)
            stages.append(SentenceDetector().setInputCols([input_col]).setOutputCol(output_col))
            input_col = output_col

        upper_case_5 = UpperCaseSparkNLPTransformer() \
            .setInputCols([output_col]) \
            .setOutputCol("upper_5")

        stages.append(upper_case_5)

        sentence_detector_5 = SentenceDetector() \
            .setInputCols(["upper_5"]) \
            .setOutputCol("sentence_5")

        stages.append(sentence_detector_5)

        pipeline = Pipeline(stages=stages)
        pipeline_model = pipeline.fit(data)
        # pipeline.fit(data).transform(data).show(truncate=False)
        light_pipeline = LightPipeline(pipeline_model)
        result = light_pipeline.annotate("light example")
        print(result)
