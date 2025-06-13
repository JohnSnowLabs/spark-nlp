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
class Phi4TransformerTextGenerationTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([
            (
                1,
                "<|start_header_id|>system<|end_header_id|> \\n" + \
                "You are a minion chatbot who always responds in minion speak! \\n" + \
                "<|start_header_id|>user<|end_header_id|> \\n" + \
                "Who are you? \\n" + \
                "<|start_header_id|>assistant<|end_header_id|> \\n"
            )
        ]).toDF("id", "text")
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        phi4 = Phi4Transformer \
            .pretrained() \
            .setMaxOutputLength(50) \
            .setDoSample(True) \
            .setBeamSize(4) \
            .setTemperature(0.6) \
            .setTopK(-1) \
            .setTopP(0.9) \
            .setStopTokenIds([128001]) \
            .setInputCols(["documents"]) \
            .setOutputCol("generation")

        pipeline = Pipeline().setStages([document_assembler, phi4])
        results = (pipeline.fit(data).transform(data))

        results.select("generation.result").show(truncate=False) 