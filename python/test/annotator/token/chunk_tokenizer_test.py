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
class ChunkTokenizerTestSpec(unittest.TestCase):

    def setUp(self):
        self.session = SparkContextForTest.spark

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        entity_extractor = TextMatcher() \
            .setInputCols(['document', 'token']) \
            .setOutputCol("entity") \
            .setEntities(path="file:///" + os.getcwd() + "/../src/test/resources/entity-extractor/test-chunks.txt")
        chunk_tokenizer = ChunkTokenizer() \
            .setInputCols(['entity']) \
            .setOutputCol('chunk_token')

        pipeline = Pipeline(stages=[document_assembler, tokenizer, entity_extractor, chunk_tokenizer])

        data = self.session.createDataFrame([
            ["Hello world, my name is Michael, I am an artist and I work at Benezar"],
            ["Robert, an engineer from Farendell, graduated last year. The other one, Lucas, graduated last week."]
        ]).toDF("text")

        pipeline.fit(data).transform(data).show()

