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
import textwrap
import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest


@pytest.mark.fast
class PragmaticSBDTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence") \
            .setCustomBounds(["%%"]) \
            .setSplitLength(235) \
            .setMinLength(4) \
            .setMaxLength(50)

        assembled = document_assembler.transform(self.data)
        sentence_detector.transform(assembled).show()


@pytest.mark.fast
class PragmaticScorerTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

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
        lemmatizer = Lemmatizer() \
            .setInputCols(["token"]) \
            .setOutputCol("lemma") \
            .setDictionary(
            path="file:///" + os.getcwd() + "/../src/test/resources/lemma-corpus-small/lemmas_small.txt",
            key_delimiter="->", value_delimiter="\t")
        sentiment_detector = SentimentDetector() \
            .setInputCols(["lemma", "sentence"]) \
            .setOutputCol("sentiment") \
            .setDictionary(
            "file:///" + os.getcwd() + "/../src/test/resources/sentiment-corpus/default-sentiment-dict.txt",
            delimiter=",")
        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.fit(sentenced).transform(sentenced)
        lemmatized = lemmatizer.fit(tokenized).transform(tokenized)
        sentiment_detector.fit(lemmatized).transform(lemmatized).show()


@pytest.mark.fast
class PragmaticSBDReturnCustomBoundsTestSpec(unittest.TestCase):

    def create_data(self, data):
        return SparkContextForTest.spark.createDataFrame([[data]]).toDF("text")

    def runTest(self):
        def assert_sentence_bounds(sent, sd, expected_sentence):
            doc_assembler = DocumentAssembler() \
                .setInputCol("text") \
                .setOutputCol("document")

            data = self.create_data(sent)
            doc = doc_assembler.transform(data)

            result = sd.transform(doc).select("sentence.result").first()["result"]

            for sent, exp in zip(result, expected_sentence):
                assert sent == exp

        example = "This is a sentence. This one uses custom bounds; As is this one;"

        sentence_detector_default = SentenceDetector() \
            .setInputCols("document") \
            .setOutputCol("sentence") \
            .setCustomBounds([r"\.", ";"]) \
            .setUseCustomBoundsOnly(True)

        expected_default = ["This is a sentence", "This one uses custom bounds",
                            "As is this one"]

        assert_sentence_bounds(example, sentence_detector_default, expected_default)

        sentence_detector = SentenceDetector() \
            .setInputCols("document") \
            .setOutputCol("sentence") \
            .setCustomBounds([r"\.", ";"]) \
            .setUseCustomBoundsOnly(True) \
            .setCustomBoundsStrategy("append")

        sentence_detector_mixed = SentenceDetector() \
            .setInputCols("document") \
            .setOutputCol("sentence") \
            .setCustomBounds([";"]) \
            .setCustomBoundsStrategy("append")

        expected_append = ["This is a sentence.", "This one uses custom bounds;",
                    "As is this one;"]

        assert_sentence_bounds(example, sentence_detector, expected_append)
        assert_sentence_bounds(example, sentence_detector_mixed, expected_append)

        subHeaderList = textwrap.dedent(
            """
            1. This is a list
            1.1 This is a subpoint
            2. Second thing
            2.2 Second subthing
            """
        )

        sentence_detector_prepend = SentenceDetector() \
            .setInputCols("document") \
            .setOutputCol("sentence") \
            .setCustomBounds([r"\n[\d\. ]+"]) \
            .setUseCustomBoundsOnly(True) \
            .setCustomBoundsStrategy("prepend")

        expectedPrepend = [
            "1. This is a list",
            "1.1 This is a subpoint",
            "2. Second thing",
            "2.2 Second subthing"]
        assert_sentence_bounds(subHeaderList, sentence_detector_prepend,
                               expectedPrepend)
