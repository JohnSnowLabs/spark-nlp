#  Copyright 2017-2022 John Snow Labs
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
class GPT2TransformerTextGenerationTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, """Leonardo Da Vinci invented the microscope?""".strip().replace("\n", " ")]]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        gpt2 = GPT2Transformer \
            .pretrained() \
            .setTask("is it true that") \
            .setMaxOutputLength(50) \
            .setDoSample(True) \
            .setInputCols(["documents"]) \
            .setOutputCol("generation")

        pipeline = Pipeline().setStages([document_assembler, gpt2])
        results = pipeline.fit(data).transform(data)

        results.select("generation.result").show(truncate=False)

