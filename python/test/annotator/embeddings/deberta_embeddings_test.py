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
class DeBertaEmbeddingsTestSpec(unittest.TestCase, HasMaxSentenceLengthTests):


    def test_run(self):
        data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        embeddings = DeBertaEmbeddings.pretrained() \
            .setInputCols(["token", "document"]) \
            .setOutputCol("camembert_embeddings")

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            embeddings
        ])

        model = pipeline.fit(data)
        model.transform(data).show()


    @pytest.mark.slow
    def test_end_to_end_pipeline(self):
        data = SparkContextForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/embeddings/sentence_embeddings.csv") \
            .limit(10)
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        embeddings =  DeBertaEmbeddings.pretrained("deberta_embeddings_erlangshen_v2_chinese_sentencepiece","zh") \
                                                               .setInputCols(["token", "document"]) \
            .setOutputCol("camembert_embeddings")


        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            embeddings
        ])

        pipeline.fit(data).transform(data).show()

