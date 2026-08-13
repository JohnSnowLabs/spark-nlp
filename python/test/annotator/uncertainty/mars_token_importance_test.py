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

# Local ONNX export of the MARS model (see examples/python/transformers/onnx for the conversion
# recipe used to produce this from https://huggingface.co/duygunuryldz/MARS).
_MARS_ONNX_PATH = "models/mars_onnx"


@pytest.mark.slow
class MarsTokenImportanceTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.data = self.spark.createDataFrame(
            [("What is the capital of France?", "Paris is the capital of France.")]
        ).toDF("question", "answer").repartition(1)

    def runTest(self):
        if not os.path.exists(_MARS_ONNX_PATH):
            self.skipTest(f"Model directory not found: {_MARS_ONNX_PATH}")

        question_assembler = DocumentAssembler().setInputCol("question").setOutputCol("question_doc")
        answer_assembler = DocumentAssembler().setInputCol("answer").setOutputCol("answer_doc")

        mars = (
            MarsTokenImportance.loadSavedModel(_MARS_ONNX_PATH, self.spark)
            .setInputCols(["question_doc", "answer_doc"])
            .setOutputCol("token_importance")
        )

        pipeline = Pipeline().setStages([question_assembler, answer_assembler, mars])
        results = pipeline.fit(self.data).transform(self.data)

        collected = results.select("token_importance").collect()
        self.assertEqual(len(collected), 1)
        annotations = collected[0]["token_importance"]
        self.assertEqual(len(annotations), 1)
        self.assertIn("token_importance", annotations[0].metadata)

        # The metadata value is a JSON array of {"begin", "end", "importance"} objects.
        import json
        phrases = json.loads(annotations[0].metadata["token_importance"])
        self.assertGreater(len(phrases), 0)
        for phrase in phrases:
            self.assertIn("begin", phrase)
            self.assertIn("end", phrase)
            self.assertIn("importance", phrase)
            self.assertTrue(0.0 <= phrase["importance"] <= 1.0)

        results.select("token_importance").show(truncate=False)
