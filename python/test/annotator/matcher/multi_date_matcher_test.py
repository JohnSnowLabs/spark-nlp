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
from pyspark.sql.functions import size
from test.util import SparkContextForTest


@pytest.mark.fast
class MultiDateMatcherTestSpec(unittest.TestCase):

    def setUp(self):
        text = """
          Lease Period	Monthly Installment of Base Rent.
          January 1, 2021 –December 31, 2021	$20,304.85 .
          January 1, 2022 –December 31, 2022	$20,914.00 .
            """
        self.data = SparkContextForTest.spark.createDataFrame([[text]]).toDF("text")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        date_matcher = MultiDateMatcher() \
            .setInputCols(["document"]) \
            .setOutputCol("date") \
            .setOutputFormat("yyyy/MM/dd") \
            .setRelaxedFactoryStrategy(MatchStrategy.MATCH_ALL)

        pipeline = Pipeline(stages=[document_assembler, date_matcher])
        model = pipeline.fit(self.data)
        result = model.transform(self.data)

        actual_dates = result.select(size("date.result")).collect()[0][0]
        self.assertEqual(actual_dates, 4)
