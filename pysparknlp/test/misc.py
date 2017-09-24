import unittest
from sparknlp.common import RegexRule


class UtilitiesTestSpec(unittest.TestCase):

    @staticmethod
    def runTest():
        regex_rule = RegexRule("\\w+", "word split")
        print(regex_rule().rule())
