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
class RoBertaForQuestionAnsweringTestSpec(unittest.TestCase, HasMaxSentenceLengthTests):
    def setUp(self):
        self.tested_annotator = RoBertaForQuestionAnswering.pretrained() \
            .setInputCols(["document_question", "document_context"]) \
            .setOutputCol("answer") \
            .setCaseSensitive(False)

    def test_run(self):
        documentAssembler = MultiDocumentAssembler() \
            .setInputCols(["question", "context"]) \
            .setOutputCols(["document_question", "document_context"])

        questionAnswering = self.tested_annotator

        pipeline = Pipeline().setStages([
            documentAssembler,
            questionAnswering
        ])

        data = SparkContextForTest.spark.createDataFrame(
            [["What's my name?", "My name is Clara and I live in Berkeley."]]).toDF("question",
                                                                                    "context")
        result = pipeline.fit(data).transform(data)
        result.select("answer.result").show(truncate=False)
