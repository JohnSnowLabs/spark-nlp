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
class MiniLMEmbeddingsTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.tested_annotator = MiniLMEmbeddings \
            .pretrained() \
            .setInputCols(["documents"]) \
            .setOutputCol("minilm")

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, "This is a sample sentence for embedding generation."],
            [2, "Another example sentence to demonstrate MiniLM embeddings."],
            [3, "MiniLM is a lightweight and efficient sentence embedding model that can generate text embeddings for various NLP tasks."],
            [4, "The model achieves comparable results with BERT-base while being much smaller and faster."]
        ]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        minilm = self.tested_annotator

        pipeline = Pipeline().setStages([document_assembler, minilm])
        results = pipeline.fit(data).transform(data)

        results.select("minilm.embeddings").show(truncate=False)
