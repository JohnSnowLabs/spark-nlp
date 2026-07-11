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
from sparknlp.common import AnnotatorType
from test.util import SparkContextForTest


@pytest.mark.fast
class SummarizationParamsTestSpec(unittest.TestCase):
    """Pure param-plumbing checks for the Python Summarization wrapper (no model download)."""

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        summ = Summarization()

        # defaults
        self.assertEqual(summ.getOrDefault(summ.method), "llm")
        self.assertEqual(summ.getOrDefault(summ.maxSummaryLength), 250)
        self.assertEqual(summ.getOrDefault(summ.minSummaryLength), 20)
        self.assertEqual(summ.getOrDefault(summ.summaryStyle), "concise")
        self.assertEqual(summ.getOrDefault(summ.longDocumentStrategy), "auto")
        self.assertEqual(summ.getOrDefault(summ.numBeams), 4)
        self.assertAlmostEqual(summ.getOrDefault(summ.mmrLambda), 0.7, places=5)
        self.assertEqual(summ.getOrDefault(summ.gpuLayers), 99)

        # annotator type contract
        self.assertEqual(Summarization.inputAnnotatorTypes, [AnnotatorType.DOCUMENT])
        self.assertEqual(Summarization.outputAnnotatorType, AnnotatorType.DOCUMENT)

        # setters round-trip
        summ = (Summarization()
                .setInputCols(["document"])
                .setOutputCol("summary")
                .setMethod("extractive")
                .setModel("custom_model")
                .setMaxSummaryLength(42)
                .setMinSummaryLength(5)
                .setSummaryStyle("bullets")
                .setFocus("main findings")
                .setLongDocumentStrategy("truncate")
                .setNumBeams(3)
                .setTemperature(0.5)
                .setTopP(0.8)
                .setChunkSize(512)
                .setChunkOverlap(2)
                .setMmrLambda(0.6)
                .setPositionBias(0.4)
                .setGpuLayers(0))

        self.assertEqual(summ.getOrDefault(summ.method), "extractive")
        self.assertEqual(summ.getOrDefault(summ.model), "custom_model")
        self.assertEqual(summ.getOrDefault(summ.maxSummaryLength), 42)
        self.assertEqual(summ.getOrDefault(summ.minSummaryLength), 5)
        self.assertEqual(summ.getOrDefault(summ.summaryStyle), "bullets")
        self.assertEqual(summ.getOrDefault(summ.focus), "main findings")
        self.assertEqual(summ.getOrDefault(summ.longDocumentStrategy), "truncate")
        self.assertEqual(summ.getOrDefault(summ.numBeams), 3)
        self.assertAlmostEqual(summ.getOrDefault(summ.temperature), 0.5, places=5)
        self.assertAlmostEqual(summ.getOrDefault(summ.topP), 0.8, places=5)
        self.assertEqual(summ.getOrDefault(summ.chunkSize), 512)
        self.assertEqual(summ.getOrDefault(summ.chunkOverlap), 2)
        self.assertAlmostEqual(summ.getOrDefault(summ.positionBias), 0.4, places=5)
        self.assertEqual(summ.getOrDefault(summ.gpuLayers), 0)

        # Summarization is an estimator; its model type is SummarizationModel
        self.assertTrue(hasattr(summ, "fit"))
        java_model = summ._java_obj  # binding exists on the JVM side
        self.assertTrue(java_model is not None)


@pytest.mark.fast
class SummarizationModelParamsTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        model = SummarizationModel()
        self.assertEqual(SummarizationModel.inputAnnotatorTypes, [AnnotatorType.DOCUMENT])
        self.assertEqual(SummarizationModel.outputAnnotatorType, AnnotatorType.DOCUMENT)
        model.setMaxSummaryLength(33)
        self.assertEqual(model.getOrDefault(model.maxSummaryLength), 33)
        # lazyAnnotator param must exist (CanBeLazy), otherwise fit()/binding breaks
        self.assertIn("lazyAnnotator", [p.name for p in model.params])
        # close() must exist and be a no-op passthrough for a non-llm delegate
        self.assertTrue(hasattr(model, "close"))
