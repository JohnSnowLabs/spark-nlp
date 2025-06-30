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
from test.util import SparkSessionForTest, SparkContextForTest

class Florence2TransformerTestSetup(unittest.TestCase):
    def setUp(self):
        self.images_path = "file://" + os.path.join(os.getcwd(), "../src/test/resources/images/")
        self.spark = SparkContextForTest.spark
        self.car_image = os.path.join(self.images_path, "car.jpg")
        self.ocr_image = os.path.join(self.images_path, "ocr_sample.jpg")
        self.spark_session = SparkSessionForTest.spark

    def build_pipeline(self, image_path, prompt):
        image_df = self.spark_session.read.format("image").load(path=image_path)
        test_df = image_df.withColumn("text", lit(prompt))
        image_assembler = ImageAssembler().setInputCol("image").setOutputCol("image_assembler")
        florence2 = Florence2Transformer \
            .pretrained() \
            .setInputCols(["image_assembler"]) \
            .setOutputCol("answer")
        pipeline = Pipeline(stages=[image_assembler, florence2])
        model = pipeline.fit(test_df)
        return model, test_df

@pytest.mark.slow
class Florence2TransformerTest(Florence2TransformerTestSetup, unittest.TestCase):
    def setUp(self):
        super().setUp()

    def run_task(self, image_path, prompt):
        model, test_df = self.build_pipeline(image_path, prompt)
        result = model.transform(test_df).collect()
        for row in result:
            self.assertTrue(row["answer"] != "")
            if "florence2_postprocessed_raw" in row["answer"][0].metadata:
                print("florence2_postprocessed_raw:", row["answer"][0].metadata["florence2_postprocessed_raw"])

    def test_ocr(self):
        self.run_task(self.ocr_image, "<OCR>")

    def test_ocr_with_region(self):
        self.run_task(self.ocr_image, "<OCR_WITH_REGION>")

    def test_captioning(self):
        for prompt in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]:
            self.run_task(self.car_image, prompt)

    def test_object_detection_and_dense_region_caption(self):
        for prompt in ["<OD>", "<DENSE_REGION_CAPTION>"]:
            self.run_task(self.car_image, prompt)

    def test_region_proposal(self):
        self.run_task(self.car_image, "<REGION_PROPOSAL>")

    def test_phrase_grounding(self):
        self.run_task(self.car_image, "<CAPTION_TO_PHRASE_GROUNDING> car")

    def test_referring_expression_segmentation(self):
        self.run_task(self.car_image, "<REFERRING_EXPRESSION_SEGMENTATION> car")

    def test_region_to_segmentation(self):
        self.run_task(self.car_image, "<REGION_TO_SEGMENTATION> region1")

    def test_open_vocabulary_detection(self):
        self.run_task(self.car_image, "<OPEN_VOCABULARY_DETECTION> car")

    def test_region_to_category(self):
        self.run_task(self.car_image, "<REGION_TO_CATEGORY> region1")

    def test_region_to_description(self):
        self.run_task(self.car_image, "<REGION_TO_DESCRIPTION> region1")

    def test_region_to_ocr(self):
        self.run_task(self.ocr_image, "<REGION_TO_OCR> region1")

@pytest.mark.slow
class LightFlorence2TransformerTest(Florence2TransformerTestSetup, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.model, self.test_df = self.build_pipeline(self.car_image, "<OD>")

    def test_light_pipeline(self):
        light_pipeline = LightPipeline(self.model)
        annotations_result = light_pipeline.fullAnnotateImage(self.car_image, "<OD>")
        for result in annotations_result:
            self.assertTrue(len(result["image_assembler"]) > 0)
            self.assertTrue(len(result["answer"]) > 0)