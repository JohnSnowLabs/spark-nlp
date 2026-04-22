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

import unittest

import pytest
from pyspark.ml import Pipeline
from sparknlp import DocumentAssembler
from sparknlp.reader import LayoutAlignerForText
from test.util import SparkContextForTest


@pytest.mark.fast
class LayoutAlignerForTextWrapperTest(unittest.TestCase):

    def setUp(self):
        spark = SparkContextForTest.spark
        # Test focused on LayoutAlignerForText input/output contracts.us
        self.input_df = spark.createDataFrame(
            [
                (
                    "Quarterly revenue increased by twenty percent.",
                    "A bar chart comparing quarterly revenue growth.",
                    "in-memory-layout-doc",
                )
            ],
            ["raw_text", "raw_caption", "fileName"],
        )

    def runTest(self):
        aligned_doc = (
            DocumentAssembler().setInputCol("raw_text").setOutputCol("aligned_doc")
        )
        image_caption = (
            DocumentAssembler().setInputCol("raw_caption").setOutputCol("image_caption")
        )

        aligner_text = LayoutAlignerForText() \
            .setInputCols(["aligned_doc", "image_caption"]) \
            .setOutputCol("aligned_text")

        pipeline = Pipeline(stages=[aligned_doc, image_caption, aligner_text])
        model = pipeline.fit(self.input_df)
        result_df = model.transform(self.input_df)

        self.assertTrue("aligned_text" in result_df.columns)
        rebuilt_text = result_df.selectExpr("aligned_text[0].result AS result").first()["result"]
        self.assertIn("Quarterly revenue increased by twenty percent.", rebuilt_text)
        self.assertIn("A bar chart comparing quarterly revenue growth.", rebuilt_text)
