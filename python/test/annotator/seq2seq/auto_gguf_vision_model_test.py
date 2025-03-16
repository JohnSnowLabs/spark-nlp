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
from test.util import SparkContextForTest


@pytest.mark.slow
class AutoGGUFVisionModelTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        documentAssembler = (
            DocumentAssembler().setInputCol("caption").setOutputCol("caption_document")
        )
        imageAssembler = (
            ImageAssembler().setInputCol("image").setOutputCol("image_assembler")
        )
        imagesPath = "../src/test/resources/image/"
        data = ImageAssembler.loadImagesAsBytes(self.spark, imagesPath).withColumn(
            "caption", lit("Caption this image.")
        )  # Add a caption to each image.
        nPredict = 40
        model = (
            AutoGGUFVisionModel.pretrained()
            .setInputCols(["caption_document", "image_assembler"])
            .setOutputCol("completions")
            .setChatTemplate("vicuna")
            .setBatchSize(4)
            .setNGpuLayers(99)
            .setNCtx(4096)
            .setMinKeep(0)
            .setMinP(0.05)
            .setNPredict(nPredict)
            .setNProbs(0)
            .setPenalizeNl(False)
            .setRepeatLastN(256)
            .setRepeatPenalty(1.18)
            .setStopStrings(["</s>", "Llama:", "User:"])
            .setTemperature(0.05)
            .setTfsZ(1)
            .setTypicalP(1)
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
            "ox.JPEG": "bull",
            "palace.JPEG": "room",
            "tractor.JPEG": "tractor",
        }

        for result in results:
            image_name = result["image_assembler"][0]["origin"].split("/")[-1]
            completion = result["completions"][0]["result"]
            assert expectedWords[image_name] in completion, f"Expected '{expectedWords[image_name]}' in '{completion}'"
