#  Copyright 2017-2025 John Snow Labs
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
import unittest

import pytest
from pyspark.sql import Row
from pyspark.sql.functions import col, explode
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from sparknlp.annotator import PairwiseVectorSimilarity
from test.util import SparkSessionForTest

# Schema for a single SENTENCE_EMBEDDINGS annotation struct.
_ANNOTATION_STRUCT = StructType([
    StructField("annotatorType", StringType(), nullable=False),
    StructField("begin", IntegerType(), nullable=False),
    StructField("end", IntegerType(), nullable=False),
    StructField("result", StringType(), nullable=False),
    StructField("metadata", MapType(StringType(), StringType(), valueContainsNull=True), nullable=True),
    StructField("embeddings", ArrayType(FloatType(), containsNull=False), nullable=False),
])

_SE_META = {"annotatorType": "sentence_embeddings"}


def _se(text, vec):
    """Build a SENTENCE_EMBEDDINGS annotation dict."""
    return {
        "annotatorType": "sentence_embeddings",
        "begin": 0,
        "end": max(len(text) - 1, 0),
        "result": text,
        "metadata": {"sentence": "0"},
        "embeddings": list(vec),
    }


def _make_df(spark, col_a_annots, col_b_annots, col_a="emb_a", col_b="emb_b"):
    schema = StructType([
        StructField(col_a, ArrayType(_ANNOTATION_STRUCT), nullable=False, metadata=_SE_META),
        StructField(col_b, ArrayType(_ANNOTATION_STRUCT), nullable=False, metadata=_SE_META),
    ])
    row = Row(**{col_a: col_a_annots, col_b: col_b_annots})
    return spark.createDataFrame([row], schema)


def _scores(df, method="cosine", col_a="emb_a", col_b="emb_b"):
    result = (
        PairwiseVectorSimilarity()
        .setInputCols([col_a, col_b])
        .setOutputCol("similarity")
        .setSimilarityMethod(method)
        .transform(df)
        .select(explode(col("similarity")).alias("s"))
        .select(col("s.result").cast("double").alias("score"))
        .collect()
    )
    return [r["score"] for r in result]


@pytest.mark.fast
class PairwiseVectorSimilarityTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSessionForTest.spark

    def test_cosine_identical(self):
        df = _make_df(self.spark,
                      [_se("hello", [1.0, 0.0, 0.0])],
                      [_se("hello", [1.0, 0.0, 0.0])])
        scores = _scores(df)
        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0], 1.0, places=9)

    def test_cosine_orthogonal(self):
        df = _make_df(self.spark,
                      [_se("q", [1.0, 0.0])],
                      [_se("d", [0.0, 1.0])])
        self.assertAlmostEqual(_scores(df)[0], 0.0, places=9)

    def test_cosine_opposite(self):
        df = _make_df(self.spark,
                      [_se("pos", [1.0, 0.0])],
                      [_se("neg", [-1.0, 0.0])])
        self.assertAlmostEqual(_scores(df)[0], -1.0, places=9)

    def test_cross_product_count(self):
        """2 embeddings in col A x 3 in col B = 6 output annotations."""
        df = _make_df(self.spark,
                      [_se("q1", [1.0, 0.0]), _se("q2", [0.0, 1.0])],
                      [_se("d1", [1.0, 0.0]), _se("d2", [0.0, 1.0]), _se("d3", [-1.0, 0.0])])
        pvs = (
            PairwiseVectorSimilarity()
            .setInputCols(["emb_a", "emb_b"])
            .setOutputCol("similarity")
        )
        count = (
            pvs.transform(df)
            .select(explode(col("similarity")).alias("s"))
            .count()
        )
        self.assertEqual(count, 6)

    def test_metadata_text_passthrough(self):
        df = _make_df(self.spark,
                      [_se("my query", [1.0, 0.0])],
                      [_se("my document", [0.0, 1.0])])
        row = (
            PairwiseVectorSimilarity()
            .setInputCols(["emb_a", "emb_b"])
            .setOutputCol("similarity")
            .transform(df)
            .select(explode(col("similarity")).alias("s"))
            .select(
                col("s.metadata")["sentence_a_text"].alias("a_text"),
                col("s.metadata")["sentence_b_text"].alias("b_text"))
            .collect()[0]
        )
        self.assertEqual(row["a_text"], "my query")
        self.assertEqual(row["b_text"], "my document")

    def test_result_equals_metadata_similarity(self):
        df = _make_df(self.spark,
                      [_se("a", [0.6, 0.8])],
                      [_se("b", [0.8, 0.6])])
        row = (
            PairwiseVectorSimilarity()
            .setInputCols(["emb_a", "emb_b"])
            .setOutputCol("similarity")
            .transform(df)
            .select(explode(col("similarity")).alias("s"))
            .select(
                col("s.result").cast("double").alias("from_result"),
                col("s.metadata")["similarity"].cast("double").alias("from_meta"))
            .collect()[0]
        )
        self.assertAlmostEqual(row["from_result"], row["from_meta"], places=15)

    def test_euclidean_identical(self):
        df = _make_df(self.spark,
                      [_se("x", [1.0, 2.0, 3.0])],
                      [_se("x", [1.0, 2.0, 3.0])])
        self.assertAlmostEqual(_scores(df, method="euclidean")[0], 0.0, places=9)

    def test_euclidean_negative_for_distinct(self):
        df = _make_df(self.spark,
                      [_se("a", [1.0, 0.0])],
                      [_se("b", [0.0, 1.0])])
        score = _scores(df, method="euclidean")[0]
        self.assertLess(score, 0.0)
        self.assertAlmostEqual(score, -math.sqrt(2.0), places=9)

    def test_dot_product(self):
        df = _make_df(self.spark,
                      [_se("a", [2.0, 3.0])],
                      [_se("b", [4.0, 5.0])])
        # 2*4 + 3*5 = 23
        self.assertAlmostEqual(_scores(df, method="dotProduct")[0], 23.0, places=9)

    def test_cosine_zero_vector(self):
        df = _make_df(self.spark,
                      [_se("zero", [0.0, 0.0])],
                      [_se("nonzero", [1.0, 0.0])])
        self.assertEqual(_scores(df)[0], 0.0)

    def test_default_method_is_cosine(self):
        df = _make_df(self.spark,
                      [_se("x", [1.0, 0.0])],
                      [_se("x", [1.0, 0.0])])
        pvs = PairwiseVectorSimilarity().setInputCols(["emb_a", "emb_b"]).setOutputCol("similarity")
        self.assertEqual(pvs.getSimilarityMethod(), "cosine")

    def test_invalid_method_raises(self):
        df = _make_df(self.spark,
                      [_se("a", [1.0, 0.0])],
                      [_se("b", [0.0, 1.0])])
        with self.assertRaisesRegex(Exception, "similarityMethod|invalid"):
            (PairwiseVectorSimilarity()
             .setInputCols(["emb_a", "emb_b"])
             .setOutputCol("similarity")
             .setSimilarityMethod("invalid")
             .transform(df)
             .collect())

    def test_set_get_similarity_method(self):
        pvs = PairwiseVectorSimilarity().setSimilarityMethod("dotProduct")
        self.assertEqual(pvs.getSimilarityMethod(), "dotProduct")
