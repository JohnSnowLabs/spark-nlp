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


@pytest.mark.fast
class SpellCheckerTestSpec(unittest.TestCase):

    def setUp(self):
        self.prediction_data = SparkContextForTest.data
        text_file = "file:///" + os.getcwd() + "/../src/test/resources/spell/sherlockholmes.txt"
        self.train_data = SparkContextForTest.spark.read.text(text_file)
        self.train_data = self.train_data.withColumnRenamed("value", "text")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        spell_checker = NorvigSweetingApproach() \
            .setInputCols(["token"]) \
            .setOutputCol("spell") \
            .setDictionary("file:///" + os.getcwd() + "/../src/test/resources/spell/words.txt")

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            spell_checker
        ])

        model = pipeline.fit(self.train_data)
        checked = model.transform(self.prediction_data)
        checked.show()

