#  Copyright 2017-2026 John Snow Labs
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
#
# from sparknlp.reader.layout_aligner_for_text import LayoutAlignerForText
# from sparknlp.reader.layout_aligner_for_vision import LayoutAlignerForVision
# from sparknlp.reader.reader_assembler import ReaderAssembler
from sparknlp.reader import ReaderAssembler, LayoutAlignerForVision, LayoutAlignerForText
from test.util import SparkContextForTest


@pytest.mark.fast
class LayoutAlignerForTextWrapperTest(unittest.TestCase):

    def setUp(self):
        spark = SparkContextForTest.spark
        self.empty_df = spark.createDataFrame([], "string").toDF("text")

    def runTest(self):
        reader = ReaderAssembler() \
            .setContentType("application/msword") \
            .setContentPath(f"file:///{os.getcwd()}/../src/test/resources/reader/doc/contains-pictures.docx") \
            .setOutputAsDocument(False) \
            .setOutputCol("data")

        aligner_vision = LayoutAlignerForVision() \
            .setInputCols(["data_text", "data_image"]) \
            .setOutputCol("aligned")

        # Use aligned_doc as both inputs here to validate the wrapper wiring and output shape
        # without requiring a multimodal captioning model in Python unit tests.
        aligner_text = LayoutAlignerForText() \
            .setInputCols(["aligned_doc", "aligned_doc"]) \
            .setOutputCol("aligned_text")

        pipeline = Pipeline(stages=[reader, aligner_vision, aligner_text])
        model = pipeline.fit(self.empty_df)
        result_df = model.transform(self.empty_df)

        self.assertTrue(result_df.count() > 0)
        self.assertTrue("aligned_text" in result_df.columns)
