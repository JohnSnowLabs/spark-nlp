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
class TypedDependencyParserConllUTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark \
            .createDataFrame([["I saw a girl with a telescope"]]).toDF("text")
        self.corpus = os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/"
        self.conllu = os.getcwd() + "/../src/test/resources/parser/unlabeled/conll-u/train_small.conllu.txt"
        self.conllu = os.getcwd() + "/../src/test/resources/parser/labeled/train_small.conllu.txt"
        from sparknlp.training import POS
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
            .setIterations(1) \
            .fit(self.train_pos)

        dependency_parser = DependencyParserApproach() \
            .setInputCols(["sentence", "pos", "token"]) \
            .setOutputCol("dependency") \
            .setConllU(self.conllu) \
            .setNumberOfIterations(10)

        typed_dependency_parser = TypedDependencyParserApproach() \
            .setInputCols(["token", "pos", "dependency"]) \
            .setOutputCol("labdep") \
            .setConllU(self.conllu) \
            .setNumberOfIterations(10)

        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.fit(sentenced).transform(sentenced)
        pos_tagged = pos_tagger.transform(tokenized)
        dependency_parsed = dependency_parser.fit(pos_tagged).transform(pos_tagged)
        typed_dependency_parsed = typed_dependency_parser.fit(dependency_parsed).transform(dependency_parsed)
        typed_dependency_parsed.show()


@pytest.mark.fast
class TypedDependencyParserConll2009TestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark \
            .createDataFrame([["I saw a girl with a telescope"]]).toDF("text")
        self.corpus = os.getcwd() + "/../src/test/resources/anc-pos-corpus-small/"
        self.tree_bank = os.getcwd() + "/../src/test/resources/parser/unlabeled/dependency_treebank"
        self.conll2009 = os.getcwd() + "/../src/test/resources/parser/labeled/example.train.conll2009"
        from sparknlp.training import POS
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
            .setIterations(1) \
            .fit(self.train_pos)

        dependency_parser = DependencyParserApproach() \
            .setInputCols(["sentence", "pos", "token"]) \
            .setOutputCol("dependency") \
            .setDependencyTreeBank(self.tree_bank) \
            .setNumberOfIterations(10)

        typed_dependency_parser = TypedDependencyParserApproach() \
            .setInputCols(["token", "pos", "dependency"]) \
            .setOutputCol("labdep") \
            .setConll2009(self.conll2009) \
            .setNumberOfIterations(10)

        assembled = document_assembler.transform(self.data)
        sentenced = sentence_detector.transform(assembled)
        tokenized = tokenizer.fit(sentenced).transform(sentenced)
        pos_tagged = pos_tagger.transform(tokenized)
        dependency_parsed = dependency_parser.fit(pos_tagged).transform(pos_tagged)
        typed_dependency_parsed = typed_dependency_parser.fit(dependency_parsed).transform(dependency_parsed)
        typed_dependency_parsed.show()

