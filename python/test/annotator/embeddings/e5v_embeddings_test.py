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

from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import lit
from test.util import SparkContextForTest

@pytest.mark.slow
class E5VEmbeddingsTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.images_path = "file://"+os.getcwd() + "/../src/test/resources/image/"

    def test_image_and_text_embedding(self):
        # Simulate image+text embedding (requires actual image files for full test)
        image_folder = os.environ.get("E5V_IMAGE_TEST_FOLDER", self.images_path)
        imagePrompt = "<|start_header_id|>user<|end_header_id|>\n\n<image>\\nSummary above image in one word: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"
        image_df = self.spark.read.format("image").option("dropInvalid", True).load(image_folder)
        test_df = image_df.withColumn("text", lit(imagePrompt))

        imageAssembler = ImageAssembler() \
            .setInputCol("image") \
            .setOutputCol("image_assembler")
        e5v = E5VEmbeddings.pretrained() \
            .setInputCols(["image_assembler"]) \
            .setOutputCol("e5v")
        pipeline = Pipeline().setStages([imageAssembler, e5v])
        results = pipeline.fit(test_df).transform(test_df)
        results.select("e5v.embeddings").show(truncate=True)

    def test_text_only_embedding(self):
        # Simulate text-only embedding using emptyImageRow and imageSchema
        from sparknlp.util import EmbeddingsDataFrameUtils
        textPrompt = "<|start_header_id|>user<|end_header_id|>\n\n<sent>\\nSummary above sentence in one word: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"
        textDesc = "A cat sitting in a box."
        nullImageDF = self.spark.createDataFrame(
            self.spark.sparkContext.parallelize([EmbeddingsDataFrameUtils.emptyImageRow]),
            EmbeddingsDataFrameUtils.imageSchema)
        textDF = nullImageDF.withColumn("text", lit(textPrompt.replace("<sent>", textDesc)))
        imageAssembler = ImageAssembler() \
            .setInputCol("image") \
            .setOutputCol("image_assembler")
        e5v = E5VEmbeddings.pretrained() \
            .setInputCols(["image_assembler"]) \
            .setOutputCol("e5v")
        pipeline = Pipeline().setStages([imageAssembler, e5v])
        results = pipeline.fit(textDF).transform(textDF)
        results.select("e5v.embeddings").show(truncate=True)