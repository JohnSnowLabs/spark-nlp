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
class StopWordsCleanerTestSpec(unittest.TestCase):
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
        stop_words_cleaner = StopWordsCleaner() \
            .setInputCols(["token"]) \
            .setOutputCol("cleanTokens") \
            .setCaseSensitive(False) \
            .setStopWords(["this", "is"])

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            stop_words_cleaner
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).select("cleanTokens.result").show()


@pytest.mark.fast
class StopWordsCleanerModelTestSpec(unittest.TestCase):
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
        stop_words_cleaner = StopWordsCleaner.pretrained() \
            .setInputCols(["token"]) \
            .setOutputCol("cleanTokens") \
            .setCaseSensitive(False)

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            stop_words_cleaner
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).select("cleanTokens.result").show()
