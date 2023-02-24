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
