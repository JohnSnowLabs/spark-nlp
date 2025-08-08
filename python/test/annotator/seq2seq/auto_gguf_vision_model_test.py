#  Copyright 2017-2023 John Snow Labs
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
from pyspark.sql.functions import lit

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkSessionForTest


@pytest.mark.slow
class AutoGGUFVisionModelTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSessionForTest.spark

    def runTest(self):
        documentAssembler = (
            DocumentAssembler().setInputCol("caption").setOutputCol("caption_document")
        )
        imageAssembler = (
            ImageAssembler().setInputCol("image").setOutputCol("image_assembler")
        )
        imagesPath = "../src/test/resources/image/"
        data = ImageAssembler.loadImagesAsBytes(self.spark, imagesPath).withColumn(
            "caption",
            lit(
                "Describe in a short and easy to understand sentence what you see in the image."
            ),
        )  # Add a caption to each image.
        nPredict = 40
        model: AutoGGUFVisionModel = (
            AutoGGUFVisionModel.pretrained()
            .setInputCols(["caption_document", "image_assembler"])
            .setOutputCol("completions")
            .setBatchSize(2)
            .setNGpuLayers(99)
            .setNCtx(4096)
            .setMinKeep(0)
            .setMinP(0.05)
            .setNPredict(nPredict)
            .setPenalizeNl(True)
            .setRepeatPenalty(1.18)
            .setTemperature(0.05)
            .setTopK(40)
            .setTopP(0.95)
        )
        pipeline = Pipeline().setStages([documentAssembler, imageAssembler, model])
        # pipeline.fit(data).transform(data).selectExpr(
        #     "reverse(split(image.origin, '/'))[0] as image_name", "completions.result"
        # ).show(truncate=False)

        results = pipeline.fit(data).transform(data).collect()

        expectedWords = {
            "bluetick.jpg": "dog",
            "chihuahua.jpg": "dog",
            "egyptian_cat.jpeg": "cat",
            "hen.JPEG": "chick",
            "hippopotamus.JPEG": "hippo",
            "junco.JPEG": "bird",
            "ostrich.JPEG": "ostrich",
            "ox.JPEG": "horn",
            "palace.JPEG": "room",
            "tractor.JPEG": "tractor",
        }

        for result in results:
            image_name = result["image_assembler"][0]["origin"].split("/")[-1]
            completion = result["completions"][0]["result"]

            print(f"Image: {image_name}, Completion: {completion}")
            assert (
                expectedWords[image_name] in completion.lower()
            ), f"Expected '{expectedWords[image_name]}' in '{completion.lower()}'"
