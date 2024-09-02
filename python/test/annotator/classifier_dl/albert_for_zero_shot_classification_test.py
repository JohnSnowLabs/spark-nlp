#  Copyright 2017-2024 John Snow Labs
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
from test.annotator.common.has_max_sentence_length_test import HasMaxSentenceLengthTests
from test.util import SparkContextForTest


@pytest.mark.slow
class AlbertForZeroShotClassificationTestSpec(unittest.TestCase, HasMaxSentenceLengthTests):
    def setUp(self):
        self.text = "I have a problem with my iphone that needs to be resolved asap!!"
        self.data = SparkContextForTest.spark \
            .createDataFrame([[self.text]]).toDF("text")
        self.candidate_labels = ["urgent", "mobile", "technology"]

        self.tested_annotator = AlbertForZeroShotClassification \
            .pretrained()\
            .setInputCols(["document", "token"]) \
            .setOutputCol("multi_class") \
            .setCandidateLabels(self.candidate_labels)

    def test_run(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        doc_classifier = self.tested_annotator

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            doc_classifier
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()

        light_pipeline = LightPipeline(model)
        annotations_result = light_pipeline.fullAnnotate(self.text)
        multi_class_result = annotations_result[0]["multi_class"][0].result
        self.assertIn(multi_class_result, self.candidate_labels)
