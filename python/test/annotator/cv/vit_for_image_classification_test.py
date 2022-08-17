#  Copyright 2017-2022 John Snow Labs
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
from test.util import SparkSessionForTest


@pytest.mark.slow
class ViTForImageClassificationTestSpec(unittest.TestCase):
    def setUp(self):
        images_path = os.getcwd() + "/../src/test/resources/image/"
        self.data = SparkSessionForTest.spark.read.format("image") \
            .load(path=images_path)

    def runTest(self):
        image_assembler = ImageAssembler() \
            .setInputCol("image") \
            .setOutputCol("image_assembler")

        imageClassifier = ViTForImageClassification \
            .pretrained() \
            .setInputCols("image_assembler") \
            .setOutputCol("class")

        pipeline = Pipeline(stages=[
            image_assembler,
            imageClassifier,
        ])

        model = pipeline.fit(self.data)
        result_df = model.transform(self.data)
        assert result_df.select("class").count() > 0


@pytest.mark.slow
class LightViTForImageClassificationOneImageTestSpec(unittest.TestCase):

    def setUp(self):
        self.images_path = os.getcwd() + "/../src/test/resources/image/"
        self.data = SparkSessionForTest.spark.read.format("image") \
            .load(path=self.images_path)

    def runTest(self):

        image_assembler = ImageAssembler() \
            .setInputCol("image") \
            .setOutputCol("image_assembler")

        image_classifier = ViTForImageClassification \
            .pretrained() \
            .setInputCols("image_assembler") \
            .setOutputCol("class")

        pipeline = Pipeline(stages=[
            image_assembler,
            image_classifier,
        ])

        model = pipeline.fit(self.data)
        light_pipeline = LightPipeline(model)
        result = light_pipeline.fullAnnotateImage(self.images_path + "hippopotamus.JPEG")

        image_assembler_result = result[0]["image_assembler"]
        class_result = result[0]["class"]
        assert len(image_assembler_result) > 0
        assert len(class_result) > 0


@pytest.mark.slow
class LightViTForImageClassificationTestSpec(unittest.TestCase):

    def setUp(self):
        self.images_path = os.getcwd() + "/../src/test/resources/image/"
        self.data = SparkSessionForTest.spark.read.format("image") \
            .load(path=self.images_path)

    def runTest(self):

        image_assembler = ImageAssembler() \
            .setInputCol("image") \
            .setOutputCol("image_assembler")

        image_classifier = ViTForImageClassification \
            .pretrained() \
            .setInputCols("image_assembler") \
            .setOutputCol("class")

        pipeline = Pipeline(stages=[
            image_assembler,
            image_classifier,
        ])

        model = pipeline.fit(self.data)
        light_pipeline = LightPipeline(model)
        images = [self.images_path + "hippopotamus.JPEG", self.images_path + "egyptian_cat.jpeg"]
        annotations_result = light_pipeline.fullAnnotateImage(images)

        for result in annotations_result:
            assert len(result["image_assembler"]) > 0
            assert len(result["class"]) > 0

