#  Copyright 2017-2022 John Snow Labs
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

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest


@pytest.mark.fast
class BasicAnnotatorsTestSpec(unittest.TestCase):

    def setUp(self):
        # This implicitly sets up py4j for us
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token") \
            .setExceptions(["New York"]) \
            .addInfixPattern("(%\\d+)")
        stemmer = Stemmer() \
            .setInputCols(["token"]) \
            .setOutputCol("stem")
        normalizer = Normalizer() \
            .setInputCols(["stem"]) \
            .setOutputCol("normalize")
        token_assembler = TokenAssembler() \
            .setInputCols(["document", "normalize"]) \
            .setOutputCol("assembled")
        finisher = Finisher() \
            .setInputCols(["assembled"]) \
            .setOutputCols(["reassembled_view"]) \
            .setCleanAnnotations(True)
        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.fit(assembled).transform(assembled)
        stemmed = stemmer.transform(tokenized)
        normalized = normalizer.fit(stemmed).transform(stemmed)
        reassembled = token_assembler.transform(normalized)
        finisher.transform(reassembled).show()

