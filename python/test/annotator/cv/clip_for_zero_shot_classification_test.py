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


class CLIPForZeroShotClassificationTestSetUp(unittest.TestCase):
    labels = [
        "a photo of a bird",
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a hen",
        "a photo of a hippo",
        "a photo of a room",
        "a photo of a tractor",
        "a photo of an ostrich",
        "a photo of an ox",
    ]
    gold_standards = {
        "bluetick.jpg": "a photo of a dog",
        "chihuahua.jpg": "a photo of a dog",
        "egyptian_cat.jpeg": "a photo of a cat",
        "hen.JPEG": "a photo of a hen",
        "hippopotamus.JPEG": "a photo of a hippo",
        "junco.JPEG": "a photo of a bird",
        "ostrich.JPEG": "a photo of an ostrich",
        "ox.JPEG": "a photo of an ox",
        "palace.JPEG": "a photo of a room",
        "tractor.JPEG": "a photo of a tractor",
    }

    def setUp(self):
        self.images_path = os.getcwd() + "/../src/test/resources/image/"
        self.data = SparkSessionForTest.spark.read.format("image").load(
            path=self.images_path
        )

        image_assembler = (
            ImageAssembler().setInputCol("image").setOutputCol("image_assembler")
        )

        imageClassifier = (
            CLIPForZeroShotClassification.pretrained()
            .setInputCols("image_assembler")
            .setOutputCol("label")
            .setCandidateLabels(self.labels)
        )

        pipeline = Pipeline(
            stages=[
                image_assembler,
                imageClassifier,
            ]
        )

        self.model = pipeline.fit(self.data)


@pytest.mark.slow
class CLIPForZeroShotClassificationTestSpec(
    CLIPForZeroShotClassificationTestSetUp, unittest.TestCase
):
    def setUp(self):
        super().setUp()

    def runTest(self):
        result = (
            self.model.transform(self.data)
            .select("image.origin", "label.result")
            .collect()
        )

        for row in result:
            file_name = row["origin"].rsplit("/", 1)[-1]
            self.assertEqual(self.gold_standards[file_name], row["result"][0])


@pytest.mark.slow
class LightCLIPForZeroShotClassificationOneImageTestSpec(
    CLIPForZeroShotClassificationTestSetUp, unittest.TestCase
):
    def setUp(self):
        super().setUp()

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        annotations_result = light_pipeline.fullAnnotateImage(
            self.images_path + "hippopotamus.JPEG"
        )

        self.assertEqual(len(annotations_result), 1)
        for result in annotations_result:
            self.assertTrue(len(result["image_assembler"]) > 0)
            self.assertTrue(len(result["label"]) > 0)


@pytest.mark.slow
class LightCLIPForZeroShotClassificationTestSpec(
    CLIPForZeroShotClassificationTestSetUp, unittest.TestCase
):
    def setUp(self):
        super().setUp()

    def runTest(self):
        light_pipeline = LightPipeline(self.model)
        images = [
            self.images_path + "hippopotamus.JPEG",
            self.images_path + "egyptian_cat.jpeg",
        ]

        annotations_result = light_pipeline.fullAnnotateImage(images)

        self.assertEqual(len(annotations_result), len(images))
        for result in annotations_result:
            self.assertTrue(len(result["image_assembler"]) > 0)
            self.assertTrue(len(result["label"]) > 0)
