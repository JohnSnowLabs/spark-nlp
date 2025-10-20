
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
from pyspark.sql.functions import explode

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

@pytest.mark.fast
class Reader2DocTestInputColumn(unittest.TestCase):

    def setUp(self):
        spark = SparkContextForTest.spark
        content = "<html><head><title>Test<title><body><p>Unclosed tag"
        self.html_df = spark.createDataFrame([(1, content)], ["id", "html"])

    def runTest(self):
        reader2doc = Reader2Doc() \
            .setInputCol("html") \
            .setOutputCol("document")

        pipeline = Pipeline(stages=[reader2doc])
        model = pipeline.fit(self.html_df)

        result_df = model.transform(self.html_df)

        self.assertTrue(result_df.select("document").count() > 0)

@pytest.mark.fast
class Reader2DocTestHierarchy(unittest.TestCase):

    def setUp(self):
        spark = SparkContextForTest.spark
        self.empty_df = spark.createDataFrame([], "string").toDF("text")

    def runTest(self):
        reader2doc = Reader2Doc() \
            .setContentType("text/html") \
            .setContentPath(f"file:///{os.getcwd()}/../src/test/resources/reader/html/simple-book.html") \
            .setOutputCol("document")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        pipeline = Pipeline(stages=[reader2doc, sentence_detector])
        model = pipeline.fit(self.empty_df)

        result_df = model.transform(self.empty_df)
        rows = result_df.select("sentence").collect()

        all_sentences = [elem for row in rows for elem in row.sentence]

        # Check for required metadata keys
        for s in all_sentences:
            metadata = s.metadata
            assert (
                    "element_id" in metadata or "parent_id" in metadata
            ), f"❌ Missing 'element_id' or 'parent_id' in metadata: {metadata}"