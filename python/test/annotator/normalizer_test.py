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
import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest


@pytest.mark.fast
class NormalizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.session = SparkContextForTest.spark
        # self.data = SparkContextForTest.data

    def runTest(self):
        data = self.session.createDataFrame([("this is some/text I wrote",)], ["text"])
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        normalizer = Normalizer() \
            .setInputCols(["token"]) \
            .setOutputCol("normalized") \
            .setLowercase(False) \
            .setMinLength(4) \
            .setMaxLength(10)

        assembled = document_assembler.transform(data)
        tokens = tokenizer.fit(assembled).transform(assembled)
        normalizer.fit(tokens).transform(tokens).show()

