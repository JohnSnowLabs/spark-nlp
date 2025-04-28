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
from pyspark.sql.functions import lit
from test.util import SparkSessionForTest
from test.util import SparkContextForTest


class Qwen2VLTransformerTestSetup(unittest.TestCase):

    def setUp(self):
        self.images_path = os.getcwd() + "/../src/test/resources/image/"
        image_df = SparkSessionForTest.spark.read.format("image").load(
            path=self.images_path
        )
        self.spark = SparkContextForTest.spark
        self.test_df = image_df.withColumn("text", lit("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"))

        image_assembler = ImageAssembler().setInputCol("image").setOutputCol("image_assembler")

        imageClassifier = Qwen2VLTransformer.pretrained() \
            .setInputCols("image_assembler") \
            .setOutputCol("answer")

        self.pipeline = Pipeline(
            stages=[
                image_assembler,
                imageClassifier,
            ]
        )

        self.model = self.pipeline.fit(self.test_df)



@pytest.mark.slow
class Qwen2VLTransformerTest(Qwen2VLTransformerTestSetup, unittest.TestCase):

   def setUp(self):
       super().setUp()

   def runTest(self):
       result = self.model.transform(self.test_df).collect()

       for row in result:
           self.assertTrue(row["answer"] != "")
           print(row["answer"])


@pytest.mark.slow
class LightQwen2VLTransformerTest(Qwen2VLTransformerTestSetup, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        light_pipeline = LightPipeline(self.model)
        image_path = self.images_path + "bluetick.jpg"
        annotations_result = light_pipeline.fullAnnotateImage(
            image_path,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"
        )

        for result in annotations_result:
            self.assertTrue(len(result["image_assembler"]) > 0)
            self.assertTrue(len(result["answer"]) > 0)
            print(result["answer"])