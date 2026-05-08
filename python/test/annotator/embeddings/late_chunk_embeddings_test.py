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

import os
import unittest

import pytest
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.functions import col as spark_col
from pyspark.sql.types import ArrayType, StructField, StructType

from sparknlp.annotator import (
    LateChunkEmbeddings,
    LongformerEmbeddings,
    NGramGenerator,
    Tokenizer,
    WordEmbeddingsModel,
)
from sparknlp.annotation import Annotation
from sparknlp.base import Doc2Chunk, DocumentAssembler, EmbeddingsFinisher
from test.util import SparkContextForTest


def _glove_pipeline(pooling="AVERAGE"):
    """DocumentAssembler → Tokenizer → GloVe → NGram chunks → LateChunkEmbeddings."""
    return Pipeline(stages=[
        DocumentAssembler().setInputCol("text").setOutputCol("document"),
        Tokenizer().setInputCols(["document"]).setOutputCol("token"),
        WordEmbeddingsModel.pretrained()
            .setInputCols(["document", "token"]).setOutputCol("embeddings")
            .setCaseSensitive(False),
        NGramGenerator().setInputCols(["token"]).setOutputCol("chunk").setN(2),
        LateChunkEmbeddings()
            .setInputCols(["document", "chunk", "embeddings"])
            .setOutputCol("late_chunk_embeddings")
            .setPoolingStrategy(pooling),
    ])


def _csv_data():
    return SparkContextForTest.spark.read \
        .option("header", "true") \
        .csv("file:///" + os.getcwd() +
             "/../src/test/resources/embeddings/sentence_embeddings.csv")


def _make_chunk_df(spark, rows):
    """Wrap raw annotation rows in a DataFrame with the NLP column metadata Spark NLP expects."""
    schema = StructType([StructField("chunk", ArrayType(Annotation.dataType()))])
    df = spark.createDataFrame([(rows,)], schema=schema)
    return df.withColumn("chunk", spark_col("chunk").alias("chunk", metadata={"annotatorType": "chunk"}))


@pytest.mark.fast
class LateChunkEmbeddingsAveragePoolingTest(unittest.TestCase):

    def runTest(self):
        data = _csv_data()
        result = _glove_pipeline("AVERAGE").fit(data).transform(data)
        rows = result.selectExpr("explode(late_chunk_embeddings) as r") \
            .select("r.annotatorType", "r.embeddings").collect()

        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertEqual(row["annotatorType"], "sentence_embeddings")
            self.assertGreater(len(row["embeddings"]), 0)

        dims = {len(r["embeddings"]) for r in rows}
        self.assertEqual(len(dims), 1, f"Inconsistent embedding dimensions: {dims}")


@pytest.mark.fast
class LateChunkEmbeddingsSumPoolingTest(unittest.TestCase):

    def runTest(self):
        data = _csv_data()
        result = _glove_pipeline("SUM").fit(data).transform(data)
        rows = result.selectExpr("explode(late_chunk_embeddings) as r") \
            .select("r.annotatorType", "r.embeddings").collect()

        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertEqual(row["annotatorType"], "sentence_embeddings")
            self.assertGreater(len(row["embeddings"]), 0)


@pytest.mark.fast
class LateChunkEmbeddingsMetadataTest(unittest.TestCase):
    """Custom CHUNK metadata (entity, confidence) must survive pooling.
    Required keys: sentence, chunk, token, pieceId, isWordStart."""

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        chunk_df = _make_chunk_df(self.spark, [
            Row(annotatorType="chunk", begin=0,  end=7,  result="AcmeDrug",
                metadata={"entity": "DRUG",      "sentence": "0", "chunk": "0", "confidence": "0.99"},
                embeddings=[]),
            Row(annotatorType="chunk", begin=25, end=32, result="migraine",
                metadata={"entity": "CONDITION", "sentence": "0", "chunk": "1", "confidence": "0.95"},
                embeddings=[]),
        ])

        text_df = self.spark.createDataFrame([("AcmeDrug was prescribed for migraine.",)], ["text"])
        text_df = Pipeline(stages=[
            DocumentAssembler().setInputCol("text").setOutputCol("document"),
            Tokenizer().setInputCols(["document"]).setOutputCol("token"),
            WordEmbeddingsModel.pretrained()
                .setInputCols(["document", "token"]).setOutputCol("embeddings")
                .setCaseSensitive(False),
        ]).fit(text_df).transform(text_df)

        lce = LateChunkEmbeddings() \
            .setInputCols(["document", "chunk", "embeddings"]) \
            .setOutputCol("late_chunk_embeddings")

        rows = lce.transform(text_df.crossJoin(chunk_df)) \
            .selectExpr("explode(late_chunk_embeddings) as r") \
            .select("r.result", "r.metadata").collect()

        self.assertGreater(len(rows), 0)
        required_keys = {"sentence", "chunk", "token", "pieceId", "isWordStart"}
        for row in rows:
            for key in required_keys:
                self.assertIn(key, row["metadata"])
            if row["result"] == "AcmeDrug":
                self.assertEqual(row["metadata"].get("entity"), "DRUG")
            if row["result"] == "migraine":
                self.assertEqual(row["metadata"].get("entity"), "CONDITION")


@pytest.mark.fast
class LateChunkEmbeddingsOutOfRangeTest(unittest.TestCase):
    """A CHUNK with no overlapping tokens is silently dropped."""

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        chunk_df = _make_chunk_df(self.spark, [
            Row(annotatorType="chunk", begin=0,   end=4,    result="Short",
                metadata={"sentence": "0", "chunk": "0"}, embeddings=[]),
            Row(annotatorType="chunk", begin=999, end=1020, result="out of range",
                metadata={"sentence": "0", "chunk": "1"}, embeddings=[]),
        ])

        text_df = self.spark.createDataFrame([("Short text here.",)], ["text"])
        text_df = Pipeline(stages=[
            DocumentAssembler().setInputCol("text").setOutputCol("document"),
            Tokenizer().setInputCols(["document"]).setOutputCol("token"),
            WordEmbeddingsModel.pretrained()
                .setInputCols(["document", "token"]).setOutputCol("embeddings")
                .setCaseSensitive(False),
        ]).fit(text_df).transform(text_df)

        lce = LateChunkEmbeddings() \
            .setInputCols(["document", "chunk", "embeddings"]) \
            .setOutputCol("late_chunk_embeddings")

        rows = lce.transform(text_df.crossJoin(chunk_df)) \
            .selectExpr("explode(late_chunk_embeddings) as r") \
            .select("r.result").collect()

        self.assertEqual(len(rows), 1, f"Expected 1 annotation, got {len(rows)}")
        self.assertEqual(rows[0]["result"], "Short")


@pytest.mark.fast
class LateChunkEmbeddingsFinisherTest(unittest.TestCase):
    """Output is consumable by EmbeddingsFinisher."""

    def runTest(self):
        data = _csv_data()
        stages = _glove_pipeline("AVERAGE").getStages() + [
            EmbeddingsFinisher()
                .setInputCols(["late_chunk_embeddings"])
                .setOutputCols(["finished_embeddings"])
                .setOutputAsVector(True)
                .setCleanAnnotations(False)
        ]
        result = Pipeline(stages=stages).fit(data).transform(data)
        self.assertGreater(result.selectExpr("explode(finished_embeddings) as e").count(), 0)


@pytest.mark.fast
class LateChunkEmbeddingsParamTest(unittest.TestCase):
    """Param round-trips; invalid strategy falls back to AVERAGE."""

    def runTest(self):
        lce = LateChunkEmbeddings()
        self.assertEqual(lce.getOrDefault("poolingStrategy"), "AVERAGE")
        self.assertTrue(lce.getOrDefault("skipOOV"))

        lce.setPoolingStrategy("SUM")
        self.assertEqual(lce.getOrDefault("poolingStrategy"), "SUM")

        lce.setSkipOOV(False)
        self.assertFalse(lce.getOrDefault("skipOOV"))

        lce.setPoolingStrategy("MAX")
        self.assertEqual(lce.getOrDefault("poolingStrategy"), "AVERAGE")


def _longformer_pipeline(extra_stages=()):
    return Pipeline(stages=[
        DocumentAssembler().setInputCol("text").setOutputCol("document"),
        Tokenizer().setInputCols(["document"]).setOutputCol("token"),
        LongformerEmbeddings.pretrained()
            .setInputCols(["document", "token"]).setOutputCol("token_embeddings")
            .setCaseSensitive(True).setMaxSentenceLength(512),
        Doc2Chunk().setInputCols(["document"]).setChunkCol("chunks")
            .setIsArray(True).setOutputCol("chunk"),
        LateChunkEmbeddings()
            .setInputCols(["document", "chunk", "token_embeddings"])
            .setOutputCol("late_chunk_embeddings")
            .setPoolingStrategy("AVERAGE"),
        *extra_stages,
    ])


@pytest.mark.slow
class LateChunkEmbeddingsLongformerEndToEndTest(unittest.TestCase):
    """Full pipeline with LongformerEmbeddings; verifies output shape and schema metadata."""

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([(
            "AcmeDrug was prescribed for migraine in March. The patient took two doses.\n\n"
            "It caused severe nausea the next day, and therapy was stopped.",
            [
                "AcmeDrug was prescribed for migraine in March. The patient took two doses.",
                "It caused severe nausea the next day, and therapy was stopped.",
            ],
        )], ["text", "chunks"])

        result = _longformer_pipeline().fit(data).transform(data)
        rows = result.selectExpr("explode(late_chunk_embeddings) as r") \
            .select("r.annotatorType", "r.result", "r.embeddings").collect()

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["annotatorType"], "sentence_embeddings")

        dims = {len(r["embeddings"]) for r in rows}
        self.assertEqual(len(dims), 1)
        dim = dims.pop()
        self.assertGreater(dim, 0)

        schema_dim = result.schema["late_chunk_embeddings"].metadata.get("dimension")
        self.assertIsNotNone(schema_dim)
        self.assertEqual(int(schema_dim), dim)

        result_texts = [r["result"] for r in rows]
        self.assertTrue(any(r.startswith("AcmeDrug") for r in result_texts))
        self.assertTrue(any(r.startswith("It caused") for r in result_texts))

        result.selectExpr("explode(late_chunk_embeddings) as r") \
            .select("r.annotatorType", "r.result", "r.embeddings").show(5, 80)


@pytest.mark.slow
class LateChunkEmbeddingsLongformerFinisherTest(unittest.TestCase):
    """Longformer → LateChunkEmbeddings → EmbeddingsFinisher produces one vector per chunk."""

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([(
            "The patient was treated with AcmeDrug. Side effects included nausea and fatigue.",
            ["The patient was treated with AcmeDrug.", "Side effects included nausea and fatigue."],
        )], ["text", "chunks"])

        finisher = EmbeddingsFinisher() \
            .setInputCols(["late_chunk_embeddings"]).setOutputCols(["finished_embeddings"]) \
            .setOutputAsVector(True).setCleanAnnotations(False)

        result = _longformer_pipeline(extra_stages=[finisher]).fit(data).transform(data)
        self.assertEqual(result.selectExpr("explode(finished_embeddings) as e").count(), 2)
