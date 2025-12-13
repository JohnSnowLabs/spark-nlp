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


@pytest.mark.local
class BigTextMatcherTestSpec(unittest.TestCase):

    @pytest.mark.slow
    def test_end_to_end_pipeline(self):
        self.spark = SparkContextForTest.spark
        data =  self.spark.read.parquet(  os.getcwd() + "/../src/test/resources/sentiment.parquet").limit(10)

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")

        entity_extractor = BigTextMatcher() \
            .setInputCols("sentence", "token") \
            .setStoragePath(
            os.getcwd() +  "/../src/test/resources/entity-extractor/test-phrases.txt",
            ReadAs.TEXT
        ) \
            .setOutputCol("entity")

        finisher = Finisher() \
            .setInputCols(["entity"]) \
            .setOutputAsArray(False) \
            .setAnnotationSplitSymbol("@") \
            .setValueSplitSymbol("#")

        recursive_pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            entity_extractor,
            finisher
        ])

        result = recursive_pipeline.fit(data).transform(data)
        result.show(truncate=False)