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

from sparknlp.training import CoNLL
from test.util import SparkContextForTest

expectedColumnNames = ["document", "sentence", "token", "pos", "label"]
expectedAnnotatorTypes = ["document", "document", "token", "pos", "named_entity"]

@pytest.mark.fast
class CoNLLTestSpec(unittest.TestCase):
    def runTest(self):
        trainingData = CoNLL().readDataset(SparkContextForTest.spark, "../src/test/resources/conll/test_conll_docid.txt")
        comparedColumnNames = list(zip(map(lambda x: x.name, trainingData.schema[1:]),expectedColumnNames))
        comparedAnnotationTypes = list(zip(map(lambda x: x.metadata["annotatorType"], trainingData.schema[1:]), expectedAnnotatorTypes))
        assert(all([x == y for x, y in (comparedColumnNames +  comparedAnnotationTypes)]))


@pytest.mark.fast
class CoNLLWithIdsTestSpec(unittest.TestCase):
    def runTest(self):
        trainingData = CoNLL(includeDocId=True).readDataset(SparkContextForTest.spark, "../src/test/resources/conll/test_conll_docid.txt")
        expectedDocIds = ["O", "1", "2", "3-1", "3-2"]
        comparedDocIds = zip(trainingData.select("doc_id").rdd.flatMap(lambda x: x).collect(), expectedDocIds)
        assert(all([x == y for x, y in comparedDocIds]))