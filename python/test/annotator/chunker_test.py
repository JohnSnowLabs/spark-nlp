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
import os
import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest


@pytest.mark.fast
class ChunkerTestSpec(unittest.TestCase):

    def setUp(self):
        from sparknlp.training import POS
        self.data = SparkContextForTest.data
        self.train_pos = POS().readDataset(SparkContextForTest.spark,
                                           os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/test-training.txt",
                                           delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                                           outputTextCol="text")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")
        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")
        pos_tagger = PerceptronApproach() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("pos") \
            .setIterations(3) \
            .fit(self.train_pos)
        chunker = Chunker() \
            .setInputCols(["sentence", "pos"]) \
            .setOutputCol("chunk") \
            .setRegexParsers(["<NNP>+", "<DT|PP\\$>?<JJ>*<NN>"])
        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.fit(sentenced).transform(sentenced)
        pos_sentence_format = pos_tagger.transform(tokenized)
        chunk_phrases = chunker.transform(pos_sentence_format)
        chunk_phrases.show()

