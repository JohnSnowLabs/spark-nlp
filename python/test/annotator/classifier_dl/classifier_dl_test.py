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
from test.util import SparkSessionForTest


@pytest.mark.slow
class ClassifierDLTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkSessionForTest.spark.read.option("header", "true") \
            .csv(path="file:///" + os.getcwd() + "/../src/test/resources/classifier/sentiment.csv")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_embeddings = UniversalSentenceEncoder.pretrained() \
            .setInputCols("document") \
            .setOutputCol("sentence_embeddings")

        classifier = ClassifierDLApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \
            .setLabelColumn("label") \
            .setRandomSeed(44)

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_embeddings,
            classifier
        ])

        model = pipeline.fit(self.data)
        model.stages[-1].write().overwrite().save('./tmp_classifierDL_model')

        classsifierdlModel = ClassifierDLModel.load("./tmp_classifierDL_model") \
            .setInputCols(["sentence_embeddings"]) \
            .setOutputCol("class")

        print(classsifierdlModel.getClasses())

