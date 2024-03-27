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
class NomicEmbeddingsTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.tested_annotator = NomicEmbeddings \
            .pretrained() \
            .setInputCols(["documents"]) \
            .setOutputCol("nomic")

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, "query: how much protein should a female eat"],
            [2, "query: summit define"],
            [3, "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 "
                "is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're "
                "expecting or training for a marathon. Check out the chart below to see how much protein you should "
                "be eating each day.", ],
            [4, "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain :"
                " the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the "
                "leaders of two or more governments."]
        ]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        nomic = self.tested_annotator

        pipeline = Pipeline().setStages([document_assembler, nomic])
        results = pipeline.fit(data).transform(data)

        results.select("nomic.embeddings").show(truncate=False)
