#  Copyright 2017-2026 John Snow Labs
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
import json
import unittest
from unittest.mock import MagicMock, patch

import pytest
from pyspark.ml import Pipeline

from sparknlp.annotator import RuleBasedMatcher, Tokenizer
from sparknlp.base import DocumentAssembler
from sparknlp.common import ReadAs
from test.util import SparkSessionForTest


@pytest.mark.fast
class RuleBasedMatcherPythonApiTestSpec(unittest.TestCase):

    def new_matcher(self):
        with patch.object(RuleBasedMatcher, "_new_java_obj", return_value=MagicMock()):
            return RuleBasedMatcher()

    def test_set_rules_accepts_python_list_and_dict(self):
        list_rules = [
            {
                "id": "noun_rule",
                "patterns": [[{"POS": "NOUN"}]],
            }
        ]
        matcher = self.new_matcher().setRules(list_rules)
        self.assertEqual(json.loads(matcher.getOrDefault(matcher.rules))[0]["id"], "noun_rule")

        dict_rule = {"id": "token_rule", "patterns": [[{"TEXT": "dogs"}]]}
        matcher = self.new_matcher().setRules(dict_rule)
        self.assertEqual(json.loads(matcher.getOrDefault(matcher.rules))["id"], "token_rule")

    def test_set_rules_accepts_json_string(self):
        rules = '[{"id":"json_rule","patterns":[[{"TEXT":"dogs"}]]}]'
        matcher = self.new_matcher().setRules(rules)
        self.assertEqual(matcher.getOrDefault(matcher.rules), rules)

    def test_set_input_cols_accepts_variable_annotation_columns(self):
        matcher = self.new_matcher().setInputCols(["document", "sentence", "token", "pos", "ner"])
        self.assertEqual(
            matcher.getOrDefault(matcher.inputCols),
            ["document", "sentence", "token", "pos", "ner"],
        )

    def test_validates_python_arguments(self):
        matcher = self.new_matcher()

        with self.assertRaises(TypeError):
            matcher.setRules(3)

        with self.assertRaises(ValueError):
            matcher.setAttributeColumns({"": "token"})

        with self.assertRaises(ValueError):
            matcher.setAlignmentMode("loose")

        with self.assertRaises(ValueError):
            matcher.setOverlapStrategy("largest")

    def test_rules_and_rules_resource_are_mutually_exclusive(self):
        matcher = self.new_matcher().setRules({"id": "x", "patterns": [[{}]]})

        with self.assertRaises(ValueError):
            matcher.setRulesResource("/tmp/rules.json", ReadAs.TEXT, {"format": "text"})


@pytest.mark.fast
class RuleBasedMatcherPythonJvmIntegrationTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSessionForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([["dogs bark"]], ["text"])
        document = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        matcher = RuleBasedMatcher() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("matches") \
            .setRules([{"id": "dogs", "patterns": [[{"TEXT": "dogs"}]]}]) \
            .setAttributeColumns({"TEXT": "token"})

        model = Pipeline(stages=[document, tokenizer, matcher]).fit(data)
        matches = model.transform(data).select("matches").first()["matches"]

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["result"], "dogs")
        self.assertEqual(matches[0]["metadata"]["rule"], "dogs")
