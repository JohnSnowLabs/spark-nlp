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
import math
import os
import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.annotator.common.has_max_sentence_length_test import HasMaxSentenceLengthTests
from test.util import SparkContextForTest


@pytest.mark.slow
class AlbertEmbeddingsTestSpec(unittest.TestCase, HasMaxSentenceLengthTests):

    def setUp(self):
        self.data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

        self.tested_annotator = AlbertEmbeddings.pretrained() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("embeddings")

    @pytest.mark.onnx
    @pytest.mark.embeddings
    @pytest.mark.text
    def test_run(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")
        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")
        albert = self.tested_annotator

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            tokenizer,
            albert
        ])

        model = pipeline.fit(self.data)
        result = model.transform(self.data)
        result.show()

        self.assertEqual(albert.getEngine(), "onnx")
        self.assertEqual(albert.getDimension(), 768)
        rows = result.select("token", "embeddings").collect()
        self.assertTrue(rows, "ALBERT pipeline must produce rows")
        for row in rows:
            self.assertTrue(row.embeddings, "ALBERT must produce token embeddings")
            self.assertEqual(len(row.embeddings), len(row.token))
            for annotation in row.embeddings:
                self.assertEqual(annotation.annotatorType, "word_embeddings")
                self.assertTrue(annotation.result)
                self.assertEqual(len(annotation.embeddings), 768)
                self.assertTrue(all(math.isfinite(value) for value in annotation.embeddings))

        origin = albert._java_obj.getClass().getProtectionDomain().getCodeSource().getLocation()
        print("ALBERT engine={}, dimension={}, rows={}, class_origin={}".format(
            albert.getEngine(), albert.getDimension(), len(rows), origin))

