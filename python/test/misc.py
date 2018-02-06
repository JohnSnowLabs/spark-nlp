import unittest
from sparknlp.common import RegexRule
from sparknlp.util import *


class UtilitiesTestSpec(unittest.TestCase):

    @staticmethod
    def runTest():
        regex_rule = RegexRule("\\w+", "word split")
        print(regex_rule.rule())


class ConfigPathTestSpec(unittest.TestCase):

    @staticmethod
    def runTest():
        assert(get_config_path() == "./application.conf")
        set_config_path("./somewhere/application.conf")
        assert(get_config_path() == "./somewhere/application.conf")
