#  Copyright 2017-2023 John Snow Labs
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

from pyspark.sql.types import StringType


@pytest.mark.fast
class Token2ChunkTestSpec(unittest.TestCase):

    def setUp(self):
        text_list = ["Hello world, this is a sentence out of nowhere", "a sentence out"]
        self.data = SparkContextForTest.spark.createDataFrame(text_list, StringType()).toDF("text")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        token2chunk = Token2Chunk() \
            .setInputCols(["token"]) \
            .setOutputCol("token_chunk")

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            token2chunk
        ])

        model = pipeline.fit(self.data)
        model.write().overwrite().save("./tmp_date2chunk_model")
        PipelineModel.load("./tmp_date2chunk_model").transform(self.data)
