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


@pytest.mark.fast
class NGramGeneratorTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.createDataFrame([
            ["This is my first sentence. This is my second."],
            ["This is my third sentence. This is my forth."]]) \
            .toDF("text").cache()

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")
        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")
        ngrams = NGramGenerator() \
            .setInputCols(["token"]) \
            .setOutputCol("ngrams") \
            .setN(2)
        ngrams_cum = NGramGenerator() \
            .setInputCols(["token"]) \
            .setOutputCol("ngrams_cum") \
            .setN(2) \
            .setEnableCumulative(True)

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            ngrams,
            ngrams_cum,
        ])

        model = pipeline.fit(self.data)
        transformed_data = model.transform(self.data)
        transformed_data.select("ngrams.result", "ngrams_cum.result").show(2, False)

        assert transformed_data.select("ngrams.result").rdd.flatMap(lambda x: x).collect() == \
               [['This is', 'is my', 'my first', 'first sentence', 'sentence .', 'This is', 'is my', 'my second',
                 'second .'],
                ['This is', 'is my', 'my third', 'third sentence', 'sentence .', 'This is', 'is my', 'my forth',
                 'forth .']]

        assert transformed_data.select("ngrams_cum.result").rdd.flatMap(lambda x: x).collect() == \
               [['This', 'is', 'my', 'first', 'sentence', '.', 'This is', 'is my', 'my first', 'first sentence',
                 'sentence .', 'This', 'is', 'my', 'second', '.', 'This is', 'is my', 'my second', 'second .'],
                ['This', 'is', 'my', 'third', 'sentence', '.', 'This is', 'is my', 'my third', 'third sentence',
                 'sentence .', 'This', 'is', 'my', 'forth', '.', 'This is', 'is my', 'my forth', 'forth .']]

