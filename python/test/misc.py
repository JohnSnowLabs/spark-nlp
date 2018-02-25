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


class ConfigPathTestSpec(unittest.TestCase):

    @staticmethod
    def runTest():
        assert(get_config_path() == "./application.conf")
        set_config_path("./somewhere/application.conf")
        assert(get_config_path() == "./somewhere/application.conf")


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

    def test_document_assembler(self):
        self.serialize_them(DocumentAssembler, "assembler")

    def test_token_assembler(self):
        self.serialize_them(TokenAssembler, "token_assembler")

    def test_finisher(self):
        self.serialize_them(Finisher, "finisher")

    def test_tokenizer(self):
        self.serialize_them(Tokenizer, "tokenizer")

    def test_stemmer(self):
        self.serialize_them(Stemmer, "stemmer")

    def test_normalizer(self):
        self.serialize_them(Normalizer, "normalizer")

    def test_regex_matcher(self):
        self.serialize_them(RegexMatcher, "regex_matcher")

    def test_lemmatizer(self):
        self.serialize_them(Lemmatizer, "lemmatizer")

    def test_date_matcher(self):
        self.serialize_them(DateMatcher, "date_matcher")

    def test_entity_extractor(self):
        self.serialize_them(EntityExtractor, "entity_extractor")

    def test_perceptron_approach(self):
        self.serialize_them(PerceptronApproach, "perceptron_approach")

    def test_sentence_detector(self):
        self.serialize_them(SentenceDetector, "sentence_detector")

    def test_sentiment_detector(self):
        self.serialize_them(SentimentDetector, "sentiment_detector")

    def test_vivekn(self):
        self.serialize_them(ViveknSentimentApproach, "vivekn")

    def test_norvig(self):
        self.serialize_them(NorvigSweetingApproach, "norvig")

    def test_ner(self):
        self.serialize_them(NerCrfApproach, "ner_crf")

    def test_assertion_log(self):
        self.serialize_them(AssertionLogRegApproach, "assertion_log")