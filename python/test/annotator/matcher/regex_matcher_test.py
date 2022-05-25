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

from sparknlp.annotator import *
from sparknlp.base import *


@pytest.mark.fast
class RegexMatcherTestSpec(unittest.TestCase):

    def setUp(self):
        # This implicitly sets up py4j for us
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        regex_matcher = RegexMatcher() \
            .setInputCols(['document']) \
            .setStrategy("MATCH_ALL") \
            .setExternalRules(path="file:///" + os.getcwd() + "/../src/test/resources/regex-matcher/rules.txt",
                              delimiter=",") \
            .setOutputCol("regex")
        assembled = document_assembler.transform(self.data)
        regex_matcher.fit(assembled).transform(assembled).show()

