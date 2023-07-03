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
from test.annotator.common.has_max_sentence_length_test import HasMaxSentenceLengthTests
from test.util import SparkSessionForTest


@pytest.mark.slow
class SpanBertCorefTestSpec(unittest.TestCase, HasMaxSentenceLengthTests):

    def setUp(self):
        self.data = SparkSessionForTest.spark.createDataFrame([
            [
                "Meanwhile Prime Minister Ehud Barak told Israeli television he doubts a peace deal can be reached before Israel''s February 6th election. He said he will now focus on suppressing Palestinian violence."],
            [
                "John loves Mary because she knows how to treat him. She is also fond of him. John said something to Mary but she didn't respond to him."],
            ["the "],
            ["  "],
            [" "]
        ]).toDF("text")

        self.tested_annotator = SpanBertCorefModel() \
            .pretrained() \
            .setInputCols(["sentences", "tokens"]) \
            .setOutputCol("corefs")

    def test_run(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentences")

        tokenizer = Tokenizer() \
            .setInputCols(["sentences"]) \
            .setOutputCol("tokens")

        coref = self.tested_annotator

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            coref
        ])

        model = pipeline.fit(self.data)

        model \
            .transform(self.data) \
            .selectExpr("explode(corefs) AS coref") \
            .selectExpr("coref.result as token", "coref.metadata") \
            .show(truncate=False)
