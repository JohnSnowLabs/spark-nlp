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
class InstructorEmbeddingsTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.tested_annotator = InstructorEmbeddings.pretrained() \
            .setInstruction("Represent the Wikipedia document for retrieval: ") \
            .setInputCols(["documents"]) \
            .setOutputCol("instructor")

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, """Capitalism has been dominant in the Western world since the end of feudalism, but most feel[who?] that 
            the term "mixed economies" more precisely describes most contemporary economies, due to their containing both 
            private-owned and state-owned enterprises. In capitalism, prices determine the demand-supply scale. For 
            example, higher demand for certain goods and services lead to higher prices and lower demand for certain 
            goods lead to lower prices. """]]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        instruction = self.tested_annotator

        pipeline = Pipeline().setStages([document_assembler, instruction])
        results = pipeline.fit(data).transform(data)

        results.select("instructor.embeddings").show(truncate=False)

#
# @pytest.mark.slow
# class BertEmbeddingsLoadSavedModelTestSpec(unittest.TestCase):
#
#     def setUp(self):
#         self.data = SparkContextForTest.spark.read.option("header", "true") \
#             .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")
#
#     def runTest(self):
#         document_assembler = DocumentAssembler() \
#             .setInputCol("text") \
#             .setOutputCol("document")
#         sentence_detector = SentenceDetector() \
#             .setInputCols(["document"]) \
#             .setOutputCol("sentence")
#         tokenizer = Tokenizer() \
#             .setInputCols(["sentence"]) \
#             .setOutputCol("token")
#         albert = BertEmbeddings.loadSavedModel(os.getcwd() + "/../src/test/resources/tf-hub-bert/model",
#                                                SparkContextForTest.spark) \
#             .setInputCols(["sentence", "token"]) \
#             .setOutputCol("embeddings")
#
#         pipeline = Pipeline(stages=[
#             document_assembler,
#             sentence_detector,
#             tokenizer,
#             albert
#         ])
#
#         model = pipeline.fit(self.data)
#         model.write().overwrite().save("./tmp_bert_pipeline_model")
#         model.transform(self.data).show()
