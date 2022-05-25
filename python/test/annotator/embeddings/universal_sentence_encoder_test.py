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
from test.util import SparkSessionForTest


@pytest.mark.slow
class UniversalSentenceEncoderTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkSessionForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")
        sentence_embeddings = UniversalSentenceEncoder.pretrained() \
            .setInputCols("sentence") \
            .setOutputCol("sentence_embeddings")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            sentence_embeddings
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()

