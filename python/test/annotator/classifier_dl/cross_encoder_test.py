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
from test.util import SparkContextForTest


@pytest.mark.slow
class CrossEncoderTestSpec(unittest.TestCase):
    def setUp(self):
        self.query = "How many people live in Berlin?"
        self.passages = [
            "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 km2.",
            "Berlin is well known for its museums.",
            "In 2014, the city state Berlin had 37,368 live births.",
        ]

    def test_run(self):
        document = MultiDocumentAssembler() \
            .setInputCols(["query", "passage"]) \
            .setOutputCols(["document1", "document2"])

        cross_encoder = CrossEncoder.pretrained() \
            .setInputCols(["document1", "document2"]) \
            .setOutputCol("score") \
            .setBatchSize(2)

        pipeline = Pipeline().setStages([document, cross_encoder])

        data = SparkContextForTest.spark.createDataFrame(
            [[self.query, passage] for passage in self.passages]
        ).toDF("query", "passage")

        result = pipeline.fit(data).transform(data)
        result.select("score.result", "score.metadata").show(truncate=False)

        rows = result.select("score").collect()
        self.assertEqual(len(rows), len(self.passages))
        for row in rows:
            self.assertEqual(len(row["score"]), 1)
            score = float(row["score"][0].result)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
