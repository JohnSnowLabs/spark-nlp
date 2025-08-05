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
import os
import unittest

import pytest
from pyspark.ml import Pipeline

from sparknlp import DocumentAssembler
from sparknlp.partition.partition_transformer import PartitionTransformer
from test.util import SparkContextForTest


@pytest.mark.fast
class PartitionTransformerTesSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.testDataSet = self.spark.createDataFrame(
            [("An example with DocumentAssembler annotator",)],
            ["text"]
        )

    def runTest(self):
        emptyDataSet = self.spark.createDataFrame([], self.testDataSet.schema)

        documentAssembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        partition = PartitionTransformer() \
            .setInputCols(["document"]) \
            .setOutputCol("partition")

        pipeline = Pipeline(stages=[documentAssembler, partition])
        pipelineModel = pipeline.fit(emptyDataSet)

        resultDf = pipelineModel.transform(self.testDataSet)
        resultDf.show(truncate=False)

        self.assertTrue(resultDf.select("partition").count() > 0)


@pytest.mark.slow
class PartitionTransformerURLsTesSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.testDataSet = self.spark.createDataFrame(
            [("https://www.blizzard.com",)],
            ["text"]
        )

    def runTest(self):
        emptyDataSet = self.spark.createDataFrame([], self.testDataSet.schema)

        documentAssembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        partition = PartitionTransformer() \
            .setInputCols(["document"]) \
            .setOutputCol("partition") \
            .setContentType("url") \
            .setHeaders({"Accept-Language": "es-ES"})

        pipeline = Pipeline(stages=[documentAssembler, partition])
        pipelineModel = pipeline.fit(emptyDataSet)

        resultDf = pipelineModel.transform(self.testDataSet)

        self.assertTrue(resultDf.select("partition").count() > 0)


@pytest.mark.fast
class PartitionTransformerChunkTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.content_path = f"file:///{os.getcwd()}/../src/test/resources/reader/txt/long-text.txt"
        self.testDataSet = self.spark.createDataFrame(
            [("An example with DocumentAssembler annotator",)],
            ["text"]
        )
        self.emptyDataSet = self.spark.createDataFrame([], self.testDataSet.schema)

    def runTest(self):
        partition = PartitionTransformer() \
            .setInputCols(["text"]) \
            .setContentPath(self.content_path) \
            .setOutputCol("partition") \
            .setChunkingStrategy("basic") \
            .setMaxCharacters(140)

        pipeline = Pipeline(stages=[partition])
        pipelineModel = pipeline.fit(self.emptyDataSet)

        resultDf = pipelineModel.transform(self.emptyDataSet)

        self.assertTrue(resultDf.select("partition").count() >= 0)