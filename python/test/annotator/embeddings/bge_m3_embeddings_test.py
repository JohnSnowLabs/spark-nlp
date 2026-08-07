#  Copyright 2017-2024 John Snow Labs
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
from pyspark.sql import functions as F
from test.util import SparkContextForTest


def _is_float(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


# Structural metadata keys added by upstream annotators / the embeddings wrapper.
_STRUCTURAL_KEYS = {"sentence", "id", "token", "pieceId", "isWordStart", "isOOV"}


def _sparse_weights(metadata):
    return {
        k: v
        for k, v in metadata.items()
        if k not in _STRUCTURAL_KEYS and _is_float(v)
    }


@pytest.mark.slow
class BGEM3EmbeddingsTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def test_multilingual_dense(self):
        data = self.spark.createDataFrame([
            [1, "How much protein should a female eat?"],
            [2, "¿Cuánta proteína debería comer una mujer?"],
            [3, "Combien de protéines une femme devrait-elle manger ?"],
            [4, "女性はどのくらいのタンパク質を摂取すべきですか？"],
        ]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        bge_m3 = BGEM3Embeddings \
            .pretrained() \
            .setInputCols(["documents"]) \
            .setOutputCol("bge_m3")

        pipeline = Pipeline().setStages([document_assembler, bge_m3])
        results = pipeline.fit(data).transform(data)

        sizes = results.select(
            F.size(results["bge_m3.embeddings"].getItem(0)).alias("size")
        ).collect()
        for row in sizes:
            self.assertEqual(row["size"], 1024)

    def test_sparse_embeddings(self):
        data = self.spark.createDataFrame(
            [["BGE-M3 supports both dense and sparse retrieval."]]
        ).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        bge_m3 = BGEM3Embeddings \
            .pretrained() \
            .setInputCols(["documents"]) \
            .setOutputCol("bge_m3") \
            .setReturnSparseEmbeddings(True)

        pipeline = Pipeline().setStages([document_assembler, bge_m3])
        results = pipeline.fit(data).transform(data)

        metadata = results.select("bge_m3.metadata").collect()[0][0][0]
        weights = _sparse_weights(metadata)
        self.assertTrue(len(weights) > 0, "Expected sparse lexical weights in metadata")
        self.assertTrue(all(float(v) > 0.0 for v in weights.values()))

    def test_long_document(self):
        long_text = " ".join(
            f"Sentence number {i} talks about multilingual retrieval and embeddings."
            for i in range(1, 300)
        )
        data = self.spark.createDataFrame([[long_text]]).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        bge_m3 = BGEM3Embeddings \
            .pretrained() \
            .setInputCols(["documents"]) \
            .setOutputCol("bge_m3") \
            .setMaxSentenceLength(8192)

        pipeline = Pipeline().setStages([document_assembler, bge_m3])
        results = pipeline.fit(data).transform(data)

        size = results.select(
            F.size(results["bge_m3.embeddings"].getItem(0)).alias("size")
        ).collect()[0]["size"]
        self.assertEqual(size, 1024)
