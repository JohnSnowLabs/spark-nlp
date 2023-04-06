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


class ConvNextForImageClassificationTestSetUp(unittest.TestCase):
    gold_standards = {
        "bluetick.jpg": "bluetick",
        "chihuahua.jpg": "Chihuahua",
        "egyptian_cat.jpeg": "tabby, tabby cat",
        "hen.JPEG": "hen",
        "hippopotamus.JPEG": "hippopotamus, hippo, river horse, Hippopotamus amphibius",
        "junco.JPEG": "junco, snowbird",
        "ostrich.JPEG": "ostrich, Struthio camelus",
        "ox.JPEG": "ox",
        "palace.JPEG": "palace",
        "tractor.JPEG": "thresher, thrasher, threshing machine",
    }

    def setUp(self):
        self.images_path = os.getcwd() + "/../src/test/resources/image/"
        self.data = SparkSessionForTest.spark.read.format("image") \
            .load(path=self.images_path)

        image_assembler = ImageAssembler() \
            .setInputCol("image") \
            .setOutputCol("image_assembler")

        imageClassifier = ConvNextForImageClassification \
            .pretrained() \
            .setInputCols("image_assembler") \
            .setOutputCol("class")

        pipeline = Pipeline(stages=[
            image_assembler,
            imageClassifier,
        ])

        self.model = pipeline.fit(self.data)


@pytest.mark.slow
class ConvNextForImageClassificationTestSpec(ConvNextForImageClassificationTestSetUp, unittest.TestCase):
    def setUp(self):
        super().setUp()

    def runTest(self):
        result = self.model.transform(self.data).select("image.origin", "class.result").collect()

        for row in result:
            file_name = row["origin"].rsplit("/", 1)[-1]
            self.assertEqual(self.gold_standards[file_name], row["result"][0])


@pytest.mark.slow
class LightConvNextForImageClassificationOneImageTestSpec(ConvNextForImageClassificationTestSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        annotations_result = light_pipeline.fullAnnotateImage(self.images_path + "hippopotamus.JPEG")

        self.assertEqual(len(annotations_result), 1)
        for result in annotations_result:
            self.assertTrue(len(result["image_assembler"]) > 0)
            self.assertTrue(len(result["class"]) > 0)


@pytest.mark.slow
class LightConvNextForImageClassificationTestSpec(ConvNextForImageClassificationTestSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        light_pipeline = LightPipeline(self.model)
        images = [self.images_path + "hippopotamus.JPEG", self.images_path + "egyptian_cat.jpeg"]

        annotations_result = light_pipeline.fullAnnotateImage(images)

        self.assertEqual(len(annotations_result), len(images))
        for result in annotations_result:
            self.assertTrue(len(result["image_assembler"]) > 0)
            self.assertTrue(len(result["class"]) > 0)
