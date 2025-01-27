#  Copyright 2017-2025 John Snow Labs
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

from sparknlp.annotator.cleaners import *
from sparknlp.base import *
from test.util import SparkContextForTest


@pytest.mark.slow
class ExtractorTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark
        eml_data = """from ABC.DEF.local ([ba23::58b5:2236:45g2:88h2]) by
  \n ABC.DEF.local2 ([ba23::58b5:2236:45g2:88h2%25]) with mapi id\
  n 32.88.5467.123; Fri, 26 Mar 2021 11:04:09 +1200"""
        self.data_set = self.spark.createDataFrame([[eml_data]]).toDF("text")

    def runTest(self):
        document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

        extractor = Extractor() \
            .setInputCols(["document"]) \
            .setOutputCol("date") \
            .setExtractorMode("email_date")

        pipeline = Pipeline().setStages([
            document_assembler,
            extractor
        ])

        model = pipeline.fit(self.data_set)
        result = model.transform(self.data_set)
        result.show(truncate=False)

