import unittest

import pytest

from sparknlp.training import SpacyToAnnotation
from test.util import SparkSessionForTest


class SpacyToAnnotationTestSetUp(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSessionForTest.spark
        self.resources_path = "../src/test/resources/spacy-to-annotation/"


@pytest.mark.fast
class SpacyToAnnotationMultiDocTestSpec(SpacyToAnnotationTestSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        nlp_reader = SpacyToAnnotation()
        result = nlp_reader.readJsonFile(self.spark, self.resources_path + "multi_doc_tokens.json")

        self.assertTrue(result.select("document").count() > 0)
        self.assertTrue(result.select("sentence").count() > 0)
        self.assertTrue(result.select("token").count() > 0)


@pytest.mark.fast
class SpacyToAnnotationWithoutSentenceTestSpec(SpacyToAnnotationTestSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        nlp_reader = SpacyToAnnotation()
        result = nlp_reader.readJsonFile(self.spark, self.resources_path + "without_sentence_ends.json")

        self.assertTrue(result.select("document").count() > 0)
        self.assertTrue(result.select("token").count() > 0)
