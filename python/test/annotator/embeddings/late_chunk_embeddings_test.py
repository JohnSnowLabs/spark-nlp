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

from sparknlp.annotator import (
    Doc2Chunk,
    EmbeddingsFinisher,
    LateChunkEmbeddings,
    LongformerEmbeddings,
    NGramGenerator,
    Tokenizer,
    WordEmbeddingsModel,
)
from sparknlp.base import DocumentAssembler
from test.util import SparkContextForTest


def _base_pipeline(pooling="AVERAGE"):
    """Returns (stages_list, late_chunk_embeddings_instance) for the CSV corpus."""
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    glove = WordEmbeddingsModel.pretrained() \
        .setInputCols(["document", "token"]) \
        .setOutputCol("embeddings") \
        .setCaseSensitive(False)

    n_grams = NGramGenerator() \
        .setInputCols(["token"]) \
        .setOutputCol("chunk") \
        .setN(2)

    lce = LateChunkEmbeddings() \
        .setInputCols(["document", "chunk", "embeddings"]) \
        .setOutputCol("late_chunk_embeddings") \
        .setPoolingStrategy(pooling)

    return [document_assembler, tokenizer, glove, n_grams, lce], lce


def _csv_data():
    return SparkContextForTest.spark.read \
        .option("header", "true") \
        .csv(path="file:///" + os.getcwd() +
             "/../src/test/resources/embeddings/sentence_embeddings.csv")


@pytest.mark.fast
class LateChunkEmbeddingsAveragePoolingTest(unittest.TestCase):
    """Output type is SENTENCE_EMBEDDINGS; embeddings are non-empty and uniform dimension."""

    def setUp(self):
        self.data = _csv_data()

    def runTest(self):
        stages, _ = _base_pipeline("AVERAGE")
        result = Pipeline(stages=stages).fit(self.data).transform(self.data)

        rows = result.selectExpr(
            "explode(late_chunk_embeddings) as r"
        ).select("r.annotatorType", "r.embeddings").collect()

        self.assertGreater(len(rows), 0, "Expected at least one chunk embedding")

        for row in rows:
            self.assertEqual(
                row["annotatorType"], "sentence_embeddings",
                f"Expected 'sentence_embeddings', got '{row['annotatorType']}'"
            )
            self.assertGreater(
                len(row["embeddings"]), 0,
                "Embedding vector must not be empty"
            )

        dims = {len(r["embeddings"]) for r in rows}
        self.assertEqual(
            len(dims), 1,
            f"All chunk embeddings must have the same dimension, got: {dims}"
        )


@pytest.mark.fast
class LateChunkEmbeddingsSumPoolingTest(unittest.TestCase):
    """SUM pooling produces non-empty SENTENCE_EMBEDDINGS without error."""

    def setUp(self):
        self.data = _csv_data()

    def runTest(self):
        stages, _ = _base_pipeline("SUM")
        result = Pipeline(stages=stages).fit(self.data).transform(self.data)

        rows = result.selectExpr(
            "explode(late_chunk_embeddings) as r"
        ).select("r.annotatorType", "r.embeddings").collect()

        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertEqual(row["annotatorType"], "sentence_embeddings")
            self.assertGreater(len(row["embeddings"]), 0)


@pytest.mark.fast
class LateChunkEmbeddingsMetadataTest(unittest.TestCase):
    """Custom CHUNK metadata keys (entity, confidence) are forwarded to the output.

    Also verifies that the required late-chunking keys (sentence, chunk, token,
    pieceId, isWordStart) are always present.
    """

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        # Build a small DataFrame with two pre-annotated CHUNK rows that carry
        # custom metadata fields which must survive the pooling step.
        from pyspark.sql import Row
        from sparknlp.annotation import Annotation

        text = "AcmeDrug was prescribed for migraine."

        # We inject the CHUNK column as a raw struct that Spark NLP can read.
        schema = Annotation.dataType()

        chunk_data = self.spark.createDataFrame(
            [([
                Row(
                    annotatorType="chunk",
                    begin=0,
                    end=7,
                    result="AcmeDrug",
                    metadata={"entity": "DRUG", "sentence": "0",
                              "chunk": "0", "confidence": "0.99"},
                    embeddings=[],
                ),
                Row(
                    annotatorType="chunk",
                    begin=25,
                    end=32,
                    result="migraine",
                    metadata={"entity": "CONDITION", "sentence": "0",
                              "chunk": "1", "confidence": "0.95"},
                    embeddings=[],
                ),
            ],)],
            ["chunk"]
        )

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        glove = WordEmbeddingsModel.pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("embeddings") \
            .setCaseSensitive(False)

        text_df = self.spark.createDataFrame([(text,)], ["text"])
        text_df = document_assembler \
            .fit(text_df).transform(text_df)
        text_df = tokenizer \
            .fit(text_df).transform(text_df)
        text_df = Pipeline(stages=[glove]) \
            .fit(text_df).transform(text_df)

        # Cross-join so every row has document + token embeddings + chunks
        full_df = text_df.crossJoin(chunk_data)

        lce = LateChunkEmbeddings() \
            .setInputCols(["document", "chunk", "embeddings"]) \
            .setOutputCol("late_chunk_embeddings") \
            .setPoolingStrategy("AVERAGE")

        result = lce.fit(full_df).transform(full_df)

        rows = result.selectExpr(
            "explode(late_chunk_embeddings) as r"
        ).select("r.result", "r.metadata").collect()

        self.assertGreater(len(rows), 0, "Expected at least one output annotation")

        required_keys = {"sentence", "chunk", "token", "pieceId", "isWordStart"}
        for row in rows:
            meta = row["metadata"]
            for key in required_keys:
                self.assertIn(
                    key, meta,
                    f"Required metadata key '{key}' missing in {meta}"
                )
            # Custom fields forwarded from input CHUNK
            if row["result"] == "AcmeDrug":
                self.assertEqual(meta.get("entity"), "DRUG")
            if row["result"] == "migraine":
                self.assertEqual(meta.get("entity"), "CONDITION")


@pytest.mark.fast
class LateChunkEmbeddingsOutOfRangeTest(unittest.TestCase):
    """A CHUNK whose span has no overlapping token embeddings is dropped without error.

    'Short' (offsets 0-4) overlaps with tokens; 'out of range' (offsets 999-1020)
    does not. Only one SENTENCE_EMBEDDINGS annotation should come out.
    """

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        from pyspark.sql import Row

        text = "Short text here."

        chunk_data = self.spark.createDataFrame(
            [([
                Row(
                    annotatorType="chunk",
                    begin=0,
                    end=4,
                    result="Short",
                    metadata={"sentence": "0", "chunk": "0"},
                    embeddings=[],
                ),
                Row(
                    annotatorType="chunk",
                    begin=999,
                    end=1020,
                    result="out of range",
                    metadata={"sentence": "0", "chunk": "1"},
                    embeddings=[],
                ),
            ],)],
            ["chunk"]
        )

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        glove = WordEmbeddingsModel.pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("embeddings") \
            .setCaseSensitive(False)

        text_df = self.spark.createDataFrame([(text,)], ["text"])
        text_df = Pipeline(stages=[document_assembler, tokenizer, glove]) \
            .fit(text_df).transform(text_df)

        full_df = text_df.crossJoin(chunk_data)

        lce = LateChunkEmbeddings() \
            .setInputCols(["document", "chunk", "embeddings"]) \
            .setOutputCol("late_chunk_embeddings") \
            .setPoolingStrategy("AVERAGE")

        result = lce.fit(full_df).transform(full_df)

        rows = result.selectExpr(
            "explode(late_chunk_embeddings) as r"
        ).select("r.result").collect()

        self.assertEqual(
            len(rows), 1,
            f"Expected 1 annotation (valid chunk only), got {len(rows)}: {[r['result'] for r in rows]}"
        )
        self.assertEqual(rows[0]["result"], "Short")


@pytest.mark.fast
class LateChunkEmbeddingsFinisherTest(unittest.TestCase):
    """LateChunkEmbeddings output can be consumed by EmbeddingsFinisher."""

    def setUp(self):
        self.data = _csv_data()

    def runTest(self):
        stages, _ = _base_pipeline("AVERAGE")

        finisher = EmbeddingsFinisher() \
            .setInputCols(["late_chunk_embeddings"]) \
            .setOutputCols(["finished_embeddings"]) \
            .setOutputAsVector(True) \
            .setCleanAnnotations(False)

        stages.append(finisher)
        result = Pipeline(stages=stages).fit(self.data).transform(self.data)

        count = result.selectExpr("explode(finished_embeddings) as e").count()
        self.assertGreater(
            count, 0,
            "EmbeddingsFinisher produced no output from LateChunkEmbeddings"
        )


@pytest.mark.fast
class LateChunkEmbeddingsParamTest(unittest.TestCase):
    """setPoolingStrategy / setSkipOOV values are readable via getters."""

    def runTest(self):
        lce = LateChunkEmbeddings()

        # Default values
        self.assertEqual(lce.getOrDefault("poolingStrategy"), "AVERAGE")
        self.assertTrue(lce.getOrDefault("skipOOV"))

        # Round-trip setters
        lce.setPoolingStrategy("SUM")
        self.assertEqual(lce.getOrDefault("poolingStrategy"), "SUM")

        lce.setSkipOOV(False)
        self.assertFalse(lce.getOrDefault("skipOOV"))

        # Invalid strategy falls back to AVERAGE
        lce.setPoolingStrategy("MAX")
        self.assertEqual(lce.getOrDefault("poolingStrategy"), "AVERAGE")


@pytest.mark.slow
class LateChunkEmbeddingsLongformerEndToEndTest(unittest.TestCase):
    """Full end-to-end pipeline: LongformerEmbeddings → Doc2Chunk → LateChunkEmbeddings.

    LongformerEmbeddings processes the entire document in a single forward pass over
    up to 4 096 tokens, which is exactly the upstream requirement for late chunking.
    Default pretrained model: ``longformer_base_4096`` (768-dim, English).

    Checks:
    - Exactly one SENTENCE_EMBEDDINGS annotation per input chunk.
    - annotatorType == ``sentence_embeddings`` on every output row.
    - All embedding vectors share the same positive dimension.
    - Schema metadata ``dimension`` matches the actual vector length.
    - Chunk result text is preserved verbatim (first token checked).
    """

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        # Two-sentence medical document. The second chunk ("It caused severe nausea...")
        # gains contextual signal from the first ("AcmeDrug was prescribed...") because
        # Longformer encodes both sentences in a single forward pass.
        data = self.spark.createDataFrame([(
            "AcmeDrug was prescribed for migraine in March. The patient took two doses.\n\n"
            "It caused severe nausea the next day, and therapy was stopped.",
            [
                "AcmeDrug was prescribed for migraine in March. The patient took two doses.",
                "It caused severe nausea the next day, and therapy was stopped.",
            ],
        )], ["text", "chunks"])

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        # LongformerEmbeddings over the FULL document — prerequisite for late chunking
        longformer = LongformerEmbeddings.pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("token_embeddings") \
            .setCaseSensitive(True) \
            .setMaxSentenceLength(512)   # kept short for test speed

        chunker = Doc2Chunk() \
            .setInputCols(["document"]) \
            .setChunkCol("chunks") \
            .setIsArray(True) \
            .setOutputCol("chunk")

        lce = LateChunkEmbeddings() \
            .setInputCols(["document", "chunk", "token_embeddings"]) \
            .setOutputCol("late_chunk_embeddings") \
            .setPoolingStrategy("AVERAGE")

        pipeline = Pipeline(stages=[
            document_assembler, tokenizer, longformer, chunker, lce
        ])

        result = pipeline.fit(data).transform(data)

        rows = result.selectExpr(
            "explode(late_chunk_embeddings) as r"
        ).select("r.annotatorType", "r.result", "r.embeddings").collect()

        # Exactly two chunks → exactly two output annotations
        self.assertEqual(
            len(rows), 2,
            f"Expected 2 annotations (one per chunk), got {len(rows)}"
        )

        # Every annotation must carry the sentence_embeddings type
        for row in rows:
            self.assertEqual(
                row["annotatorType"], "sentence_embeddings",
                f"Expected 'sentence_embeddings', got '{row['annotatorType']}'"
            )

        # Embeddings must be non-empty and uniform dimension
        dims = {len(row["embeddings"]) for row in rows}
        self.assertEqual(len(dims), 1, f"All embeddings should share one dimension, got {dims}")
        dim = dims.pop()
        self.assertGreater(dim, 0, "Embedding dimension must be positive")

        # Schema metadata dimension must match the actual vector length
        schema_dim = result.schema["late_chunk_embeddings"].metadata.get("dimension")
        self.assertIsNotNone(schema_dim, "Schema metadata 'dimension' key is missing")
        self.assertEqual(
            int(schema_dim), dim,
            f"Schema metadata dimension ({schema_dim}) != actual vector length ({dim})"
        )

        # Chunk text must be preserved verbatim
        results = [row["result"] for row in rows]
        self.assertTrue(
            any(r.startswith("AcmeDrug") for r in results),
            f"Expected first chunk to start with 'AcmeDrug', got: {results}"
        )
        self.assertTrue(
            any(r.startswith("It caused") for r in results),
            f"Expected second chunk to start with 'It caused', got: {results}"
        )

        result.selectExpr("explode(late_chunk_embeddings) as r") \
            .select("r.annotatorType", "r.result", "r.embeddings") \
            .show(5, 80)


@pytest.mark.slow
class LateChunkEmbeddingsLongformerFinisherTest(unittest.TestCase):
    """LateChunkEmbeddings (fed by Longformer) can be consumed by EmbeddingsFinisher.

    Verifies the full retrieval-ready pipeline:
    Longformer → LateChunkEmbeddings → EmbeddingsFinisher → dense vector column.
    """

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([(
            "The patient was treated with AcmeDrug. Side effects included nausea and fatigue.",
            [
                "The patient was treated with AcmeDrug.",
                "Side effects included nausea and fatigue.",
            ],
        )], ["text", "chunks"])

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        longformer = LongformerEmbeddings.pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("token_embeddings") \
            .setCaseSensitive(True) \
            .setMaxSentenceLength(512)

        chunker = Doc2Chunk() \
            .setInputCols(["document"]) \
            .setChunkCol("chunks") \
            .setIsArray(True) \
            .setOutputCol("chunk")

        lce = LateChunkEmbeddings() \
            .setInputCols(["document", "chunk", "token_embeddings"]) \
            .setOutputCol("late_chunk_embeddings") \
            .setPoolingStrategy("AVERAGE")

        finisher = EmbeddingsFinisher() \
            .setInputCols(["late_chunk_embeddings"]) \
            .setOutputCols(["finished_embeddings"]) \
            .setOutputAsVector(True) \
            .setCleanAnnotations(False)

        pipeline = Pipeline(stages=[
            document_assembler, tokenizer, longformer, chunker, lce, finisher
        ])

        result = pipeline.fit(data).transform(data)

        count = result.selectExpr("explode(finished_embeddings) as e").count()
        self.assertEqual(
            count, 2,
            f"Expected 2 finished embedding vectors (one per chunk), got {count}"
        )


