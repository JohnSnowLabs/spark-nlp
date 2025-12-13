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
class XlmRoBertaForTokenClassificationTestSpec(unittest.TestCase, HasMaxSentenceLengthTests):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.data = self.spark.createDataFrame([
            ("John Lenon was born in London and lived in Paris. My name is Sarah and I live in London",),
            ("Rare Hendrix song draft sells for almost $17,000.",),
            ("EU rejects German call to boycott British lamb.",)
        ]).toDF("text")

        self.tested_annotator = XlmRoBertaForTokenClassification \
            .pretrained("xlm_roberta_base_token_classifier_conll03") \
            .setInputCols(["document", "token"]) \
            .setOutputCol("ner")

    def test_run(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        token_classifier = self.tested_annotator

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            token_classifier
        ])

        model = pipeline.fit(self.data)
        model.transform(self.data).show()

    @pytest.mark.slow
    def test_end_to_end_pipeline(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")


        tokenizer = Tokenizer() \
            .setInputCols("document") \
            .setOutputCol("token")

        token_classifier = self.tested_annotator

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            token_classifier
        ])

        pipeline.fit(self.data).transform(self.data).show()

