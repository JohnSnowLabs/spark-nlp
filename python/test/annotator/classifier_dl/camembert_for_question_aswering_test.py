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
import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest


@pytest.mark.slow
class CamemBertForQuestionAnsweringTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.question = "Où est-ce que je vis?"
        self.context = "Mon nom est Wolfgang et je vis à Berlin"
        self.inputDataset = self.spark.createDataFrame([[self.question, self.context]]) \
            .toDF("question", "context")

    def runTest(self):
        document_assembler = MultiDocumentAssembler() \
            .setInputCols("question", "context") \
            .setOutputCols("document_question", "document_context")

        qa_classifier = CamemBertForQuestionAnswering.pretrained() \
            .setInputCols("document_question", "document_context") \
            .setOutputCol("answer") \
            .setCaseSensitive(True) \
            .setMaxSentenceLength(512)

        pipeline = Pipeline(stages=[
            document_assembler,
            qa_classifier
        ])

        model = pipeline.fit(self.inputDataset)
        model.transform(self.inputDataset).show()
        light_pipeline = LightPipeline(model)
        annotations_result = light_pipeline.fullAnnotate(self.question, self.context)
