#  Copyright 2017-2026 John Snow Labs
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
from test.util import *

# Local GGUF model used by these tests, mirroring the AutoGGUFModelTest.scala convention.
_GGUF_MODEL_PATH = "models/Qwen3-1.7B-Q4_K_M.gguf"


@pytest.mark.slow
class LLMUncertaintyEstimatorEmbeddingsBackendTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.data = self.spark.createDataFrame(
            [["What is the capital of France?"]]
        ).toDF("text").repartition(1)

    def runTest(self):
        if not os.path.exists(_GGUF_MODEL_PATH):
            self.skipTest(f"Model file not found: {_GGUF_MODEL_PATH}")

        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

        llm = (
            AutoGGUFModel.loadSavedModel(_GGUF_MODEL_PATH, self.spark)
            .setInputCols("document")
            .setOutputCol("completions")
            .setNumSamples(5)
            .setTemperature(0.7)
            .setNPredict(20)
        )

        embeddings = (
            E5Embeddings.pretrained()
            .setInputCols("completions")
            .setOutputCol("sample_embeddings")
        )

        uncertainty = (
            LLMUncertaintyEstimator()
            .setInputCols(["completions", "sample_embeddings"])
            .setOutputCol("uncertainty")
            .setMethods(["semanticEntropy"])
        )

        pipeline = Pipeline().setStages([document_assembler, llm, embeddings, uncertainty])
        results = pipeline.fit(self.data).transform(self.data)

        collected = results.select("uncertainty").collect()
        self.assertEqual(len(collected), 1)
        annotations = collected[0]["uncertainty"]
        self.assertEqual(len(annotations), 1)
        metadata = annotations[0].metadata
        self.assertIn("uncertainty_score", metadata)
        self.assertIn("confidence_score", metadata)
        self.assertIn("semantic_entropy", metadata)
        self.assertIn("num_semantic_clusters", metadata)

        results.select("uncertainty").show(truncate=False)


@pytest.mark.slow
class LLMUncertaintyEstimatorWhiteBoxTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.data = self.spark.createDataFrame(
            [["What is the capital of France?"]]
        ).toDF("text").repartition(1)

    def runTest(self):
        if not os.path.exists(_GGUF_MODEL_PATH):
            self.skipTest(f"Model file not found: {_GGUF_MODEL_PATH}")

        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

        llm = (
            AutoGGUFModel.loadSavedModel(_GGUF_MODEL_PATH, self.spark)
            .setInputCols("document")
            .setOutputCol("completions")
            .setOutputLogProbs(True)
            .setNProbs(5)
            .setNPredict(20)
        )

        uncertainty = (
            LLMUncertaintyEstimator()
            .setInputCols(["completions"])
            .setOutputCol("uncertainty")
            .setMethods(["meanLogProb", "perplexity"])
        )

        pipeline = Pipeline().setStages([document_assembler, llm, uncertainty])
        results = pipeline.fit(self.data).transform(self.data)

        collected = results.select("uncertainty").collect()
        annotations = collected[0]["uncertainty"]
        metadata = annotations[0].metadata
        self.assertIn("mean_log_prob", metadata)
        self.assertIn("perplexity", metadata)
        # perplexity = exp(-meanLogProb) must be >= 1
        self.assertGreaterEqual(float(metadata["perplexity"]), 1.0)

        results.select("uncertainty").show(truncate=False)
