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

    def test_mixed_length_batch_does_not_misalign_sparse_weights(self):
        data = self.spark.createDataFrame([
            ["Hi."],
            ["BGE-M3 supports both dense and sparse retrieval across many languages and "
             "very long documents that stretch on for a while to make padding meaningfully "
             "different across rows in the batch."],
        ]).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        bge_m3 = BGEM3Embeddings \
            .pretrained() \
            .setInputCols(["documents"]) \
            .setOutputCol("bge_m3") \
            .setReturnSparseEmbeddings(True) \
            .setBatchSize(2)

        pipeline = Pipeline().setStages([document_assembler, bge_m3])
        results = pipeline.fit(data).transform(data).select("bge_m3.metadata").collect()

        short_weights = _sparse_weights(results[0][0][0])
        long_weights = _sparse_weights(results[1][0][0])
        self.assertTrue(len(short_weights) > 0)
        self.assertTrue(len(long_weights) > 0)
        self.assertLess(len(short_weights), len(long_weights))

    def test_mixed_language_batch_in_one_call(self):
        data = self.spark.createDataFrame([
            ["How much protein should a female eat?"],
            ["¿Cuánta proteína debería comer una mujer?"],
            ["女性はどのくらいのタンパク質を摂取すべきですか？"],
            ["امرأة كم من البروتين يجب أن تأكل؟"],
        ]).toDF("text")

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
        self.assertEqual(len(sizes), 4)
        for row in sizes:
            self.assertEqual(row["size"], 1024)

    def test_batch_size_edge_cases(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        single_row = self.spark.createDataFrame(
            [["A single sentence in its own batch."]]
        ).toDF("text")
        single_bge_m3 = BGEM3Embeddings \
            .pretrained() \
            .setInputCols(["documents"]) \
            .setOutputCol("bge_m3") \
            .setBatchSize(1)
        single_pipeline = Pipeline().setStages([document_assembler, single_bge_m3])
        single_size = single_pipeline.fit(single_row).transform(single_row).select(
            F.size(F.col("bge_m3.embeddings").getItem(0)).alias("size")
        ).collect()[0]["size"]
        self.assertEqual(single_size, 1024)

        few_rows = self.spark.createDataFrame(
            [["One."], ["Two."], ["Three."]]
        ).toDF("text")
        few_bge_m3 = BGEM3Embeddings \
            .pretrained() \
            .setInputCols(["documents"]) \
            .setOutputCol("bge_m3") \
            .setBatchSize(8)  # larger than the number of rows
        few_pipeline = Pipeline().setStages([document_assembler, few_bge_m3])
        few_sizes = few_pipeline.fit(few_rows).transform(few_rows).select(
            F.size(F.col("bge_m3.embeddings").getItem(0)).alias("size")
        ).collect()
        self.assertEqual(len(few_sizes), 3)
        for row in few_sizes:
            self.assertEqual(row["size"], 1024)

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
