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

from sparknlp.annotator.cleaners import *
from sparknlp.base import *
from test.util import SparkContextForTest


@pytest.mark.fast
class CleanerBytesTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark
        eml_data = """Hello ð\x9f\x98\x80"""
        self.data_set = self.spark.createDataFrame([[eml_data]]).toDF("text")

    def runTest(self):
        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

        cleaner = Cleaner() \
            .setInputCols(["document"]) \
            .setOutputCol("cleaned") \
            .setCleanerMode("bytes_string_to_string")

        pipeline = Pipeline().setStages([
            document_assembler,
            cleaner
        ])

        model = pipeline.fit(self.data_set)
        result = model.transform(self.data_set)
        result.show(truncate=False)

@pytest.mark.fast
class CleanerBulletsTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark
        data = [("1.1 This is a very important point",),
                ("a.1 This is a very important point",),
                ("1.4.2 This is a very important point",)]
        self.data_set = self.spark.createDataFrame(data).toDF("text")

    def runTest(self):
        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

        cleaner = Cleaner() \
            .setInputCols(["document"]) \
            .setOutputCol("cleaned") \
            .setCleanerMode("clean_ordered_bullets")

        pipeline = Pipeline().setStages([
            document_assembler,
            cleaner
        ])

        model = pipeline.fit(self.data_set)
        result = model.transform(self.data_set)
        result.show(truncate=False)