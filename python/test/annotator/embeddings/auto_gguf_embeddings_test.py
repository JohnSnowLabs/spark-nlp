#  Copyright 2017-2023 John Snow Labs
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
class AutoGGUFModelTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.data = (
            self.spark.createDataFrame(
                [
                    ["The moons of Jupiter are "],
                    ["Earth is "],
                    ["The moon is "],
                    ["The sun is "],
                ]
            )
            .toDF("text")
            .repartition(1)
        )
        self.document_assembler = (
            DocumentAssembler().setInputCol("text").setOutputCol("document")
        )

    def runTest(self):
        model = (
            AutoGGUFEmbeddings.pretrained()
            .setInputCols("document")
            .setOutputCol("embeddings")
            .setBatchSize(4)
            .setNGpuLayers(99)
        )

        pipeline = Pipeline().setStages([self.document_assembler, model])
        results = pipeline.fit(self.data).transform(self.data)
        collected = results.select("embeddings.embeddings").collect()

        for row in collected:
            embds = row["embeddings"][0]
            assert embds is not None
            assert (
                sum(embds) > 0
            ), "Embeddings should not be zero. Was there an error on llama.cpp side?"


@pytest.mark.slow
class AutoGGUFEmbeddingsPoolingTypeTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.data = (
            self.spark.createDataFrame(
                [
                    ["The moons of Jupiter are "],
                    ["Earth is "],
                    ["The moon is "],
                    ["The sun is "],
                ]
            )
            .toDF("text")
            .repartition(1)
        )
        self.document_assembler = (
            DocumentAssembler().setInputCol("text").setOutputCol("document")
        )

    def runTest(self):
        model = (
            # AutoGGUFEmbeddings.pretrained()
            AutoGGUFEmbeddings.loadSavedModel(
                "models/nomic-embed-text-v1.5.Q8_0.gguf", SparkContextForTest.spark
            )
            .setInputCols("document")
            .setOutputCol("embeddings")
            .setBatchSize(4)
            .setNGpuLayers(99)
            .setPoolingType("CLS")
        )

        pipeline = Pipeline().setStages([self.document_assembler, model])
        results = pipeline.fit(self.data).transform(self.data)
        collected = results.select("embeddings.embeddings").collect()

        for row in collected:
            embds = row["embeddings"][0]
            assert embds is not None
            assert (
                sum(embds) > 0
            ), "Embeddings should not be zero. Was there an error on llama.cpp side?"
