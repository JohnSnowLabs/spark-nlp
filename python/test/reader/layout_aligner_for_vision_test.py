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
from pyspark.sql import functions as F

from sparknlp import DocumentAssembler
from sparknlp.base import ImageAssembler
from sparknlp.reader.layout_aligner_for_vision import LayoutAlignerForVision
from test.util import SparkContextForTest


@pytest.mark.fast
class LayoutAlignerForVisionWrapperTest(unittest.TestCase):

    def setUp(self):
        spark = SparkContextForTest.spark
        # It validates LayoutAlignerForVision I/O contracts on deterministic local data.
        images_df = spark.read.format("image").load(
            path=f"file:///{os.getcwd()}/../src/test/resources/image/"
        )
        self.input_df = images_df.limit(1).withColumn(
            "text", F.lit("Quarterly revenue increased compared to last year.")
        )

    def runTest(self):
        document_assembler = (
            DocumentAssembler().setInputCol("text").setOutputCol("data_text")
        )
        image_assembler = (
            ImageAssembler().setInputCol("image").setOutputCol("data_image")
        )

        aligner = LayoutAlignerForVision() \
            .setInputCols(["data_text", "data_image"]) \
            .setOutputCol("aligned") \
            .setExplodeDocs(False) \
            .setImageCaptionBasePrompt("Provide a concise financial caption for this image") \
            .setNeighborTextCharsWindow(20)

        pipeline = Pipeline(stages=[document_assembler, image_assembler, aligner])
        model = pipeline.fit(self.input_df)
        result_df = model.transform(self.input_df)

        self.assertTrue(result_df.count() > 0)
        self.assertTrue("aligned_doc" in result_df.columns)
        self.assertTrue("aligned_image" in result_df.columns)
        self.assertTrue("aligned_prompt" in result_df.columns)
