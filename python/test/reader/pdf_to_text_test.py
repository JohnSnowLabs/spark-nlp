
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

from sparknlp.reader.pdf_to_text import PdfToText
from test.util import SparkContextForTest
from pyspark.ml import Pipeline


class PdfToTextTestSetup(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.spark.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")

@pytest.mark.slow
class PdfToTextTest(PdfToTextTestSetup, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.pdf_to_text = PdfToText().setStoreSplittedPdf(True)
        pdf_path = os.getcwd() + "/../src/test/resources/reader/pdf"
        self.data_frame = self.spark.read.format("binaryFile").load(pdf_path)

    def test_run(self):
        pipeline = Pipeline(stages=[self.pdf_to_text])
        pipeline_model = pipeline.fit(self.data_frame)
        pdf_df = pipeline_model.transform(self.data_frame)
        pdf_df.show()

        self.assertTrue(pdf_df.count() > 0)


