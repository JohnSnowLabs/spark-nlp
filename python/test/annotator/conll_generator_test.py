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

from test.util import SparkContextForTest
from sparknlp.util import CoNLLGenerator


@pytest.mark.fast
class CoNLLGeneratorTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkContextForTest.spark.read.load("file:///" + os.getcwd() + "/../src/test/resources/conllgenerator/conllgenerator_nonint_token_metadata.parquet").cache()

    def runTest(self):
        CoNLLGenerator.exportConllFiles(self.data, './tmp_noninttokens2')  # with sentence
        CoNLLGenerator.exportConllFiles(self.data, './tmp_noninttokensst3', 'section')  # with section

