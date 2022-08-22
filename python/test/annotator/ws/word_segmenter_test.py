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
class WordSegmenterTestSpec(unittest.TestCase):

    def setUp(self):
        from sparknlp.training import POS
        self.data = SparkContextForTest.spark.createDataFrame([["十四不是四十"]]) \
            .toDF("text").cache()
        self.train = POS().readDataset(SparkContextForTest.spark,
                                       os.getcwd() + "/../src/test/resources/word-segmenter/chinese_train.utf8",
                                       delimiter="|", outputPosCol="tags", outputDocumentCol="document",
                                       outputTextCol="text")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        word_segmenter = WordSegmenterApproach() \
            .setInputCols("document") \
            .setOutputCol("token") \
            .setPosColumn("tags") \
            .setNIterations(1) \
            .fit(self.train)
        pipeline = Pipeline(stages=[
            document_assembler,
            word_segmenter
        ])

        model = pipeline.fit(self.train)
        model.transform(self.data).show(truncate=False)

