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

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest


@pytest.mark.slow
class DistilBertForZeroShotClassificationTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.text = "I have a problem with my iphone that needs to be resolved asap!!"
        self.inputDataset = self.spark.createDataFrame([[self.text]]) \
            .toDF("text")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        zero_shot_classifier = DistilBertForZeroShotClassification \
            .pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("class") \
            .setCandidateLabels(["urgent", "mobile", "travel", "movie", "music", "sport", "weather", "technology"])

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            zero_shot_classifier
        ])

        model = pipeline.fit(self.inputDataset)
        model.transform(self.inputDataset).show()
        light_pipeline = LightPipeline(model)
        annotations_result = light_pipeline.fullAnnotate(self.text)
