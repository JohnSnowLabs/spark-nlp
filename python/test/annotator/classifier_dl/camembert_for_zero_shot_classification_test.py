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
import os
import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.annotator.common.has_max_sentence_length_test import HasMaxSentenceLengthTests
from test.util import SparkContextForTest


@pytest.mark.slow
class CamemBertForZeroShotClassificationTestSpec(unittest.TestCase, HasMaxSentenceLengthTests):
    def setUp(self):
        self.text = "L'équipe de France joue aujourd'hui au Parc des Princes"
        self.data = SparkContextForTest.spark \
            .createDataFrame([[self.text]]).toDF("text")

        self.tested_annotator = CamemBertForZeroShotClassification \
            .pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("class")

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
        print(annotations_result)
