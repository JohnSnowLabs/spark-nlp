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
class MPNetEmbeddingsTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.tested_annotator = MPNetEmbeddings \
            .pretrained()\
            .setInputCols(["documents"]) \
            .setOutputCol("mpnet_embeddings")

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, "This is an example sentence"],
            [2,  "Each sentence is converted"],
        ]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        e5 = self.tested_annotator

        pipeline = Pipeline().setStages([document_assembler, e5])
        results = pipeline.fit(data).transform(data)

        results.select("mpnet_embeddings.embeddings").show(truncate=False)
