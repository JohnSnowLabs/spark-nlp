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
class Word2VecTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.spark.createDataFrame([
            ["Rare Hendrix song draft sells for almost $17,000. This is my second sentenece! The third one here!"],
            ["EU rejects German call to boycott British lamb ."],
            ["TORONTO 1996-08-21"],
            [" carbon emissions have come down without impinging on our growth . . ."],
            ["carbon emissions have come down without impinging on our growth .\\u2009.\\u2009."],
            ["the "],
            ["  "],
            [" "]
        ]).toDF("text")

    def runTest(self):
        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
        tokenizer = Tokenizer().setInputCols("document").setOutputCol("token")
        doc2vec = Word2VecApproach() \
            .setInputCols(["token"]) \
            .setOutputCol("sentence_embeddings") \
            .setMaxSentenceLength(512) \
            .setStepSize(0.025) \
            .setMinCount(5) \
            .setVectorSize(300) \
            .setNumPartitions(1) \
            .setMaxIter(2) \
            .setSeed(42) \
            .setEnableCaching(True) \
            .setStorageRef("doc2vec_aclImdb")

        pipeline = Pipeline(stages=[document_assembler, tokenizer, doc2vec])
        model = pipeline.fit(self.data)
        model.write().overwrite().save("./tmp_model")
        loaded_model = model.load("./tmp_model")
        loaded_model.transform(self.data).show()

