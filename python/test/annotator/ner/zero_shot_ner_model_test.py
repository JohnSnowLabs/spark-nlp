#  Copyright 2017-2023 John Snow Labs
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
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from test.util import SparkContextForTest


@pytest.mark.slow
class ZeroShotNerTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame(
            [["My name is Clara, I live in New York and Hellen lives in Paris."]]
        ).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")

        zero_shot_ner = ZeroShotNerModel() \
            .pretrained("roberta_base_qa_squad2") \
            .setEntityDefinitions(
            {
                "NAME": ["What is his name?", "What is my name?", "What is her name?"],
                "CITY": ["Which city?", "Which is the city?"]
            }) \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("zero_shot_ner")

        results = Pipeline() \
            .setStages([document_assembler, sentence_detector, tokenizer, zero_shot_ner]) \
            .fit(data) \
            .transform(data) \
            .cache()

        results \
            .selectExpr("document", "explode(filter(zero_shot_ner, x -> x.result <> \"O\")) AS entity") \
            .select("document.result",
                    "entity.result",
                    "entity.metadata.word",
                    "entity.metadata.confidence",
                    "entity.metadata.question") \
            .show(truncate=False)

        self.assertEqual(
            results
            .selectExpr("explode(zero_shot_ner) AS entity")
            .filter("entity.result <> \"O\"")
            .count(), 5)
