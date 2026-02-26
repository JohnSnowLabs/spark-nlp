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
import unittest

import pytest

from sparknlp.annotator import *
from test.util import SparkSessionForTest


@pytest.mark.fast
class NerDLApproachTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSessionForTest.spark

    def test_setters(self):
        ner_approach = (
            NerDLApproach()
            .setLr(0.01)
            .setPo(0.005)
            .setBatchSize(16)
            .setDropout(0.01)
            .setGraphFolder("graph_folder")
            .setConfigProtoBytes([])
            .setUseContrib(False)
            .setEnableMemoryOptimizer(True)
            .setIncludeConfidence(True)
            .setIncludeAllConfidenceScores(True)
            .setUseBestModel(True)
            .setPrefetchBatches(20)
            .setOptimizePartitioning(True)
        )

        # Check param map
        param_map = ner_approach.extractParamMap()
        self.assertEqual(param_map[ner_approach.lr], 0.01)
        self.assertEqual(param_map[ner_approach.po], 0.005)
        self.assertEqual(param_map[ner_approach.batchSize], 16)
        self.assertEqual(param_map[ner_approach.dropout], 0.01)
        self.assertEqual(param_map[ner_approach.graphFolder], "graph_folder")
        self.assertEqual(param_map[ner_approach.configProtoBytes], [])
        self.assertEqual(param_map[ner_approach.useContrib], False)
        self.assertEqual(param_map[ner_approach.enableMemoryOptimizer], True)
        self.assertEqual(param_map[ner_approach.includeConfidence], True)
        self.assertEqual(param_map[ner_approach.includeAllConfidenceScores], True)
        self.assertEqual(param_map[ner_approach.useBestModel], True)
        self.assertEqual(param_map[ner_approach.prefetchBatches], 20)
        self.assertEqual(param_map[ner_approach.optimizePartitioning], True)
