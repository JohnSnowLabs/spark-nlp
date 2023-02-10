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