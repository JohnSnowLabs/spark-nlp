
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
import os

from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.reader.reader2table import Reader2Table
from test.util import SparkContextForTest
from pyspark.ml import Pipeline

@pytest.mark.fast
class Reader2TableTest(unittest.TestCase):
    def setUp(self):
        spark = SparkContextForTest.spark
        self.empty_df = spark.createDataFrame([], "string").toDF("text")

    def runTest(self):
        reader2table = Reader2Table() \
            .setContentType("text/html") \
            .setContentPath(f"file:///{os.getcwd()}/../src/test/resources/reader/html/example-mix-tags.html") \
            .setOutputCol("document")

        pipeline = Pipeline(stages=[reader2table])
        model = pipeline.fit(self.empty_df)

        result_df = model.transform(self.empty_df)
        result_df.show(truncate=False)

        self.assertTrue(result_df.select("document").count() > 0)

@pytest.mark.fast
class Reader2TableMixedFilesTest(unittest.TestCase):
    def setUp(self):
        spark = SparkContextForTest.spark
        self.empty_df = spark.createDataFrame([], "string").toDF("text")

    def runTest(self):
        reader2table = Reader2Table() \
            .setContentPath(f"{os.getcwd()}/../src/test/resources/reader") \
            .setOutputCol("document")

        pipeline = Pipeline(stages=[reader2table])
        model = pipeline.fit(self.empty_df)

        result_df = model.transform(self.empty_df)

        self.assertTrue(result_df.select("document").count() > 1)