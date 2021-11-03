#  Copyright 2017-2021 John Snow Labs
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
import shutil
import tempfile

from sparknlp.common import RegexRule
from sparknlp.util import *

from sparknlp.base import *
from sparknlp.annotator import *


class UtilitiesTestSpec(unittest.TestCase):

    @staticmethod
    def runTest():
        regex_rule = RegexRule("\w+", "word split")
        assert(regex_rule.rule() == "\w+")


class SerializersTestSpec(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def serialize_them(self, cls, dirname):
        f = self.test_dir + dirname
        c1 = cls()
        c1.save(f)
        c2 = cls().load(f)
        assert(c1.uid == c2.uid)

    def runTest(self):
        self.serialize_them(DocumentAssembler, "assembler")
        self.serialize_them(TokenAssembler, "token_assembler")
        self.serialize_them(Finisher, "finisher")
        self.serialize_them(Tokenizer, "tokenizer")
        self.serialize_them(Stemmer, "stemmer")
        self.serialize_them(Normalizer, "normalizer")
        self.serialize_them(RegexMatcher, "regex_matcher")
        self.serialize_them(Lemmatizer, "lemmatizer")
        self.serialize_them(DateMatcher, "date_matcher")
        self.serialize_them(TextMatcher, "entity_extractor")
        self.serialize_them(PerceptronApproach, "perceptron_approach")
        self.serialize_them(SentenceDetector, "sentence_detector")
        self.serialize_them(SentimentDetector, "sentiment_detector")
        self.serialize_them(ViveknSentimentApproach, "vivekn")
        self.serialize_them(NorvigSweetingApproach, "norvig")
        self.serialize_them(NerCrfApproach, "ner_crf")
