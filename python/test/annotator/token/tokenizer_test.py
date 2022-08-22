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
class TokenizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.session = SparkContextForTest.spark

    def runTest(self):
        data = self.session.createDataFrame([("this is some/text I wrote",)], ["text"])
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token") \
            .addInfixPattern("(\\p{L}+)(\\/)(\\p{L}+\\b)") \
            .setMinLength(3) \
            .setMaxLength(6)
        finisher = Finisher() \
            .setInputCols(["token"]) \
            .setOutputCols(["token_out"]) \
            .setOutputAsArray(True)
        assembled = document_assembler.transform(data)
        tokenized = tokenizer.fit(assembled).transform(assembled)
        finished = finisher.transform(tokenized)
        print(finished.first()['token_out'])
        self.assertEqual(len(finished.first()['token_out']), 4)


@pytest.mark.fast
class TokenizerWithExceptionsTestSpec(unittest.TestCase):

    def setUp(self):
        self.session = SparkContextForTest.spark

    def runTest(self):
        data = self.session.createDataFrame(
            [("My friend moved to New York. She likes it. Frank visited New York, and didn't like it.",)], ["text"])
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token") \
            .setExceptionsPath(path="file:///" + os.getcwd() + "/../src/test/resources/token_exception_list.txt")
        finisher = Finisher() \
            .setInputCols(["token"]) \
            .setOutputCols(["token_out"]) \
            .setOutputAsArray(True)
        assembled = document_assembler.transform(data)
        tokenized = tokenizer.fit(assembled).transform(assembled)
        finished = finisher.transform(tokenized)
        # print(finished.first()['token_out'])
        self.assertEqual((finished.first()['token_out']).index("New York."), 4)
        self.assertEqual((finished.first()['token_out']).index("New York,"), 11)

