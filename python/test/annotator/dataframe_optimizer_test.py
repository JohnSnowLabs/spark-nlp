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
from pyspark.ml import Pipeline

from sparknlp import DocumentAssembler
from sparknlp.annotator import SentenceDetector
from sparknlp.annotator.dataframe_optimizer import DataFrameOptimizer
from test.util import SparkSessionForTest

@pytest.mark.fast
class DataframeOptimizerTestSpec(unittest.TestCase):

    def setUp(self):
        spark = SparkSessionForTest.spark
        self.test_df = spark.createDataFrame([
            (1, "test"),
            (2, "example")
        ], ["id", "text"])

    def runTest(self):

        optimizer = DataFrameOptimizer() \
            .setNumPartitions(2) \
            .setDoCache(True) \

        optimized_df = optimizer.transform(self.test_df)
        optimized_df.show()

        self.assertEqual(optimized_df.rdd.getNumPartitions(), 2)
        self.assertTrue(optimized_df.is_cached)

@pytest.mark.fast
class DataframeOptimizerPipelineTestSpec(unittest.TestCase):

    def setUp(self):
        spark = SparkSessionForTest.spark
        self.test_df = spark.createDataFrame([
            ("This is a test sentence. It contains multiple sentences.",)
        ], ["text"])

    def runTest(self):
        executor_cores = 4
        num_workers = 2

        data_frame_optimizer = DataFrameOptimizer() \
            .setExecutorCores(executor_cores) \
            .setNumWorkers(num_workers) \
            .setDoCache(True)

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentences")

        pipeline = Pipeline(stages=[
            data_frame_optimizer,
            document_assembler,
            sentence_detector
        ])

        optimized_result_df = pipeline.fit(self.test_df).transform(self.test_df)

        expected_partitions = executor_cores * num_workers
        self.assertEqual(optimized_result_df.rdd.getNumPartitions(), expected_partitions)
