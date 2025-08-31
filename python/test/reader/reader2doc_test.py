
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

import os
import unittest

import pytest
from pyspark.ml import Pipeline

from sparknlp.annotator import *
from sparknlp.reader.reader2doc import Reader2Doc
from test.util import SparkContextForTest


@pytest.mark.fast
class Reader2DocTest(unittest.TestCase):

    def setUp(self):
        spark = SparkContextForTest.spark
        self.empty_df = spark.createDataFrame([], "string").toDF("text")

    def runTest(self):
        reader2doc = Reader2Doc() \
            .setContentType("text/html") \
            .setContentPath(f"file:///{os.getcwd()}/../src/test/resources/reader/html/title-test.html") \
            .setOutputCol("document")

        pipeline = Pipeline(stages=[reader2doc])
        model = pipeline.fit(self.empty_df)

        result_df = model.transform(self.empty_df)

        self.assertTrue(result_df.select("document").count() > 0)


@pytest.mark.fast
class Reader2DocTokenTest(unittest.TestCase):

    def setUp(self):
        spark = SparkContextForTest.spark
        self.empty_df = spark.createDataFrame([], "string").toDF("text")

    def runTest(self):
        reader2doc = Reader2Doc() \
            .setContentType("text/html") \
            .setContentPath(f"file:///{os.getcwd()}/../src/test/resources/reader/html/example-div.html") \
            .setOutputCol("document") \
            .setTitleThreshold(18.5)

        regex_tok = RegexTokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("regex_token")

        pipeline = Pipeline(stages=[reader2doc, regex_tok])
        model = pipeline.fit(self.empty_df)

        result_df = model.transform(self.empty_df)

        self.assertTrue(result_df.select("document").count() > 0)


@pytest.mark.fast
class Reader2DocPdfTest(unittest.TestCase):

    def setUp(self):
        spark = SparkContextForTest.spark
        self.empty_df = spark.createDataFrame([], "string").toDF("text")

    def runTest(self):
        reader2doc = Reader2Doc() \
            .setContentType("application/pdf") \
            .setContentPath(f"file:///{os.getcwd()}/../src/test/resources/reader/pdf/pdf-title.pdf") \
            .setOutputCol("document") \
            .setTitleThreshold(18.5)

        pipeline = Pipeline(stages=[reader2doc])
        model = pipeline.fit(self.empty_df)

        result_df = model.transform(self.empty_df)

        self.assertTrue(result_df.select("document").count() > 0)

@pytest.mark.fast
class Reader2DocTestOutputAsDoc(unittest.TestCase):

    def setUp(self):
        spark = SparkContextForTest.spark
        self.empty_df = spark.createDataFrame([], "string").toDF("text")

    def runTest(self):
        reader2doc = Reader2Doc() \
            .setContentType("text/html") \
            .setContentPath(f"file:///{os.getcwd()}/../src/test/resources/reader/html/title-test.html") \
            .setOutputCol("document") \
            .setOutputAsDocument(True)

        pipeline = Pipeline(stages=[reader2doc])
        model = pipeline.fit(self.empty_df)

        result_df = model.transform(self.empty_df)

        self.assertTrue(result_df.select("document").count() > 0)