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

from sparknlp.annotator import Qwen2VLTransformer
from sparknlp.reader.reader2image import Reader2Image
from test.util import SparkContextForTest


@pytest.mark.fast
class Reader2ImageTest(unittest.TestCase):

    def setUp(self):
        spark = SparkContextForTest.spark
        self.empty_df = spark.createDataFrame([], "string").toDF("text")

    def runTest(self):
        reader2image = Reader2Image() \
            .setContentType("text/html") \
            .setContentPath(f"file:///{os.getcwd()}/../src/test/resources/reader/html/example-images.html") \
            .setOutputCol("image")

        pipeline = Pipeline(stages=[reader2image])
        model = pipeline.fit(self.empty_df)

        result_df = model.transform(self.empty_df)

        self.assertTrue(result_df.select("image").count() > 0)


@pytest.mark.slow
class Reader2ImageIntegrationVLMTest(unittest.TestCase):

    def setUp(self):
        spark = SparkContextForTest.spark
        self.empty_df = spark.createDataFrame([], "string").toDF("text")
        self.files_dir = f"{os.getcwd()}/../src/test/resources/reader/email"

    def runTest(self):
        reader2image = Reader2Image() \
            .setContentPath(self.files_dir) \
            .setOutputCol("image")

        pipeline = Pipeline(stages=[reader2image])
        model = pipeline.fit(self.empty_df)
        images_df = model.transform(self.empty_df)

        images_df.show()

        visual_qa = Qwen2VLTransformer.pretrained() \
            .setInputCols(["image"]) \
            .setOutputCol("answer")

        vlm_pipeline = Pipeline(stages=[visual_qa])
        result_df = vlm_pipeline.fit(images_df).transform(images_df)

        result_df.select("image.origin", "answer.result").show(truncate=False)

        # Assertion
        self.assertTrue(result_df.count() > 0)