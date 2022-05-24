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
from sparknlp.training import CoNLLU
from test.util import SparkContextForTest


@pytest.mark.fast
class LemmatizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        lemmatizer = Lemmatizer() \
            .setInputCols(["token"]) \
            .setOutputCol("lemma") \
            .setDictionary(path="file:///" + os.getcwd() + "/../src/test/resources/lemma-corpus-small/lemmas_small.txt",
                           key_delimiter="->", value_delimiter="\t")
        assembled = document_assembler.transform(self.data)
        tokenized = tokenizer.fit(assembled).transform(assembled)
        lemmatizer.fit(tokenized).transform(tokenized).show()


@pytest.mark.fast
class LemmatizerWithTrainingDataSetTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.conllu_file = "file:///" + os.getcwd() + "/../src/test/resources/conllu/en.test.lemma.conllu"

    def runTest(self):
        test_dataset = self.spark.createDataFrame([["So what happened?"]]).toDF("text")
        train_dataset = CoNLLU(lemmaCol="lemma_train").readDataset(self.spark, self.conllu_file)
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        lemmatizer = Lemmatizer() \
            .setInputCols(["token"]) \
            .setFormCol("form") \
            .setLemmaCol("lemma_train") \
            .setOutputCol("lemma")

        train_dataset.show()
        pipeline = Pipeline(stages=[document_assembler, tokenizer, lemmatizer])
        pipeline.fit(train_dataset).transform(test_dataset).show()

