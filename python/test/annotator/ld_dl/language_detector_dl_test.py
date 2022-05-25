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
class LanguageDetectorDLTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark.read \
            .option("delimiter", "|") \
            .option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/language-detector/multilingual_sample.txt")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetectorDLModel.pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        ld = LanguageDetectorDL.pretrained()

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            ld
        ])

        # list all the languages
        print(ld.getLanguages())

        model = pipeline.fit(self.data)
        model.transform(self.data).show()

