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
from test.util import SparkContextForTest


@pytest.mark.slow
class MPNetForSequenceClassificationTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.createDataFrame(
            [
                ["I love driving my car."],
                ["The next bus will arrive in 20 minutes."],
                ["pineapple on pizza is the worst 🤮"],
            ]
        ).toDF("text")

        self.tested_annotator = (
            MPNetForSequenceClassification.pretrained()
            .setInputCols(["document", "token"])
            .setOutputCol("label")
            .setBatchSize(8)
            .setMaxSentenceLength(384)
            .setCaseSensitive(False)
        )

    def test_run(self):
        document_assembler = (
            DocumentAssembler().setInputCol("text").setOutputCol("document")
        )

        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")

        MPNet = self.tested_annotator

        pipeline = Pipeline(stages=[document_assembler, tokenizer, MPNet])

        model = pipeline.fit(self.data)
        model.transform(self.data).select("label.result").show()
