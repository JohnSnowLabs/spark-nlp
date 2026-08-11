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
import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import *

# The published NLI checkpoint, written in this annotator's own serialization format. The
# MarsTokenImportance tests cover the loadSavedModel path instead, so both loaders stay exercised.
_NLI_MODEL = "bert_base_uncased_mnli_entailment_onnx"


@pytest.mark.slow
class SampleEntailmentMatrixTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        # Three sampled "completions" for one row: two paraphrases of the same answer, one
        # different answer - simulating AutoGGUFModel.setNumSamples(3) output.
        self.data = (
            self.spark.createDataFrame([[["Paris.", "The capital is Paris.", "London."]]])
            .toDF("completions_raw")
        )

    def runTest(self):
        document_assembler = (
            DocumentAssembler().setInputCol("completions_raw").setOutputCol("document")
        )

        entailment = (
            SampleEntailmentMatrix.pretrained(_NLI_MODEL, "en")
            .setInputCols(["document"])
            .setOutputCol("entailment")
        )

        pipeline = Pipeline().setStages([document_assembler, entailment])
        results = pipeline.fit(self.data).transform(self.data)

        collected = results.select("entailment").collect()
        annotations = collected[0]["entailment"]
        self.assertEqual(len(annotations), 3)

        import json
        # The matrix describes the row as a whole, so it is written once rather than copied onto
        # all N samples; LLMUncertaintyEstimator looks it up across the row's completions.
        carrying = [a for a in annotations if "entailment_matrix" in a.metadata]
        self.assertEqual(len(carrying), 1)

        matrix = json.loads(annotations[0].metadata["entailment_matrix"])
        self.assertEqual(len(matrix), 3)
        for row in matrix:
            self.assertEqual(len(row), 3)
        for i in range(3):
            self.assertAlmostEqual(matrix[i][i], 1.0, places=5)

        results.select("entailment").show(truncate=False)


@pytest.mark.slow
class SampleEntailmentMatrixMaxSamplesTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        # 11 samples > default maxSamplesForNli (10) must raise a clear error, not silently do
        # 110 pairwise NLI calls.
        many_samples = [f"Answer number {i}." for i in range(11)]
        data = self.spark.createDataFrame([[many_samples]]).toDF("completions_raw")

        document_assembler = (
            DocumentAssembler().setInputCol("completions_raw").setOutputCol("document")
        )
        entailment = (
            SampleEntailmentMatrix.pretrained(_NLI_MODEL, "en")
            .setInputCols(["document"])
            .setOutputCol("entailment")
        )
        pipeline = Pipeline().setStages([document_assembler, entailment])

        with self.assertRaises(Exception):
            pipeline.fit(data).transform(data).collect()
