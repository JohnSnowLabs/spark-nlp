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


@pytest.mark.fast
class LLMUncertaintyEstimatorParamsTestSpec(unittest.TestCase):
    """Param plumbing and validation, with no model or generation involved.

    Worth covering from Python specifically: pyspark writes param values straight onto the Java
    object without going through the Scala setters, so the Scala-side ``require``s never run for a
    Python-configured pipeline. These setters have to catch bad input themselves.
    """

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def test_defaults(self):
        estimator = LLMUncertaintyEstimator()
        self.assertEqual(estimator.getMethods(), ["semanticEntropy"])
        self.assertEqual(estimator.getSimilarityBackend(), "embeddings")
        self.assertAlmostEqual(estimator.getSimilarityThreshold(), 0.85, places=5)
        self.assertAlmostEqual(estimator.getEntailmentThreshold(), 0.5, places=5)
        self.assertAlmostEqual(estimator.getEigenThreshold(), 0.9, places=5)
        self.assertFalse(estimator.getEnsemble())

    def test_unset_optional_getters_return_none(self):
        estimator = LLMUncertaintyEstimator()
        self.assertIsNone(estimator.getThreshold())
        self.assertIsNone(estimator.getEnsembleWeights())

    def test_set_and_get_roundtrip(self):
        estimator = (
            LLMUncertaintyEstimator()
            .setMethods(["meanLogProb", "perplexity"])
            .setSimilarityBackend("nli")
            .setSimilarityThreshold(0.7)
            .setEntailmentThreshold(0.6)
            .setEigenThreshold(0.8)
            .setEnsemble(True)
            .setEnsembleWeights([1.0, 3.0])
            .setThreshold(0.42)
        )
        self.assertEqual(estimator.getMethods(), ["meanLogProb", "perplexity"])
        self.assertEqual(estimator.getSimilarityBackend(), "nli")
        self.assertAlmostEqual(estimator.getSimilarityThreshold(), 0.7, places=5)
        self.assertAlmostEqual(estimator.getEntailmentThreshold(), 0.6, places=5)
        self.assertAlmostEqual(estimator.getEigenThreshold(), 0.8, places=5)
        self.assertTrue(estimator.getEnsemble())
        self.assertEqual(estimator.getEnsembleWeights(), [1.0, 3.0])
        self.assertAlmostEqual(estimator.getThreshold(), 0.42, places=5)

    def test_rejects_unknown_method(self):
        with self.assertRaises(ValueError) as raised:
            LLMUncertaintyEstimator().setMethods(["semanticEntropy", "notAMethod"])
        self.assertIn("notAMethod", str(raised.exception))

    def test_rejects_empty_methods(self):
        with self.assertRaises(ValueError):
            LLMUncertaintyEstimator().setMethods([])

    def test_rejects_repeated_methods(self):
        with self.assertRaises(ValueError) as raised:
            LLMUncertaintyEstimator().setMethods(["mars", "perplexity", "mars"])
        self.assertIn("repeat", str(raised.exception))

    def test_rejects_unknown_similarity_backend(self):
        with self.assertRaises(ValueError) as raised:
            LLMUncertaintyEstimator().setSimilarityBackend("cosine")
        self.assertIn("cosine", str(raised.exception))

    def test_rejects_mismatched_ensemble_weights(self):
        estimator = LLMUncertaintyEstimator().setMethods(["meanLogProb", "perplexity"])
        with self.assertRaises(ValueError) as raised:
            estimator.setEnsembleWeights([1.0, 2.0, 3.0])
        self.assertIn("ensembleWeights", str(raised.exception))

    def test_rejects_negative_ensemble_weights(self):
        estimator = LLMUncertaintyEstimator().setMethods(["meanLogProb", "perplexity"])
        with self.assertRaises(ValueError):
            estimator.setEnsembleWeights([-1.0, 2.0])

    def test_rejects_zero_sum_ensemble_weights(self):
        estimator = LLMUncertaintyEstimator().setMethods(["meanLogProb", "perplexity"])
        with self.assertRaises(ValueError):
            estimator.setEnsembleWeights([0.0, 0.0])

    def test_accepts_every_supported_method(self):
        estimator = LLMUncertaintyEstimator().setMethods(
            LLMUncertaintyEstimator.SUPPORTED_METHODS
        )
        self.assertEqual(
            estimator.getMethods(), LLMUncertaintyEstimator.SUPPORTED_METHODS
        )


@pytest.mark.fast
class AutoGGUFModelUncertaintyParamsTestSpec(unittest.TestCase):
    """The two params AutoGGUFModel gained for uncertainty estimation, without loading a model."""

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def test_defaults(self):
        model = AutoGGUFModel()
        self.assertEqual(model.getNumSamples(), 1)
        self.assertFalse(model.getOutputLogProbs())

    def test_set_and_get_roundtrip(self):
        model = AutoGGUFModel().setNumSamples(5).setOutputLogProbs(True)
        self.assertEqual(model.getNumSamples(), 5)
        self.assertTrue(model.getOutputLogProbs())
