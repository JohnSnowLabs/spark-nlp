import unittest

import pytest

from sparknlp.common.properties import HasMaxSentenceLengthLimit


class HasMaxSentenceLengthTests:
    tested_annotator = None
    valid_max_length = 512
    over_max_length = 5000

    def test_max_length(self):
        if not self.tested_annotator:
            raise Exception("Please set the annotator to \"tested_annotator\" before running this test.")

        self.tested_annotator.setMaxSentenceLength(self.valid_max_length)

        with pytest.raises(ValueError):
            self.tested_annotator.setMaxSentenceLength(self.over_max_length)


@pytest.mark.fast
class HasMaxSentenceLengthTestSpec(unittest.TestCase, HasMaxSentenceLengthTests):
    def setUp(self):
        class MockAnnotator(HasMaxSentenceLengthLimit):
            def _set(self, maxSentenceLength):
                pass

        self.tested_annotator = MockAnnotator()
