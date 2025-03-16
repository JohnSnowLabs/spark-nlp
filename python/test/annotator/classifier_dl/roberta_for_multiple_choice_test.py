#  Copyright 2017-2025 John Snow Labs
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

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest


class RobertaForMultipleChoiceTestSetup(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.question = "The Eiffel Tower is located in which country?"
        self.choices = "Germany, France, Italy"

        self.spark = SparkContextForTest.spark
        empty_df = self.spark.createDataFrame([[""]]).toDF("text")

        document_assembler = MultiDocumentAssembler() \
            .setInputCols(["question", "context"]) \
            .setOutputCols(["document_question", "document_context"])

        model_path = "/media/danilo/Data/Danilo/JSL/models/transformers/spark-nlp/onnx/roberta_multiple_choice"
        roberta_for_multiple_choice = RoBertaForMultipleChoice.load(model_path) \
            .setInputCols(["document_question", "document_context"]) \
            .setOutputCol("answer")

        pipeline = Pipeline(stages=[document_assembler, roberta_for_multiple_choice])

        self.pipeline_model = pipeline.fit(empty_df)


@pytest.mark.slow
class RobertaForMultipleChoiceTest(RobertaForMultipleChoiceTestSetup, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.data = self.spark.createDataFrame([[self.question, self.choices]]).toDF("question","context")
        self.data.show(truncate=False)

    def test_run(self):
        result_df = self.pipeline_model.transform(self.data)
        result_df.show(truncate=False)
        for row in result_df.collect():
            self.assertTrue(row["answer"][0].result != "")


@pytest.mark.slow
class LightRobertaForMultipleChoiceTest(RobertaForMultipleChoiceTestSetup, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        light_pipeline = LightPipeline(self.pipeline_model)
        annotations_result = light_pipeline.fullAnnotate(self.question,self.choices)
        print(annotations_result)
        for result in annotations_result:
            self.assertTrue(result["answer"][0].result != "")

        result = light_pipeline.annotate(self.question,self.choices)
        print(result)
        self.assertTrue(result["answer"] != "")
