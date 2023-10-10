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


@pytest.mark.fast
class Date2ChunkTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        date_matcher = DateMatcher() \
            .setInputCols(['document']) \
            .setOutputCol("date") \
            .setOutputFormat("yyyyMM") \
            .setSourceLanguage("en")

        date_to_chunk = Date2Chunk() \
            .setInputCols(['date']) \
            .setOutputCol("date_chunk") \
            .setEntityName("DATUM")

        pipeline = Pipeline(stages=[
            document_assembler,
            date_matcher,
            date_to_chunk
        ])

        model = pipeline.fit(self.data)
        model.write().overwrite().save("./tmp_date2chunk_model")
        PipelineModel.load("./tmp_date2chunk_model").transform(self.data)
