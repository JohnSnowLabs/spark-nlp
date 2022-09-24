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


@pytest.mark.slow
class TapasForQuestionAnsweringTestSpec(unittest.TestCase):
    def setUp(self):
        with open(os.getcwd() + "/../src/test/resources/tapas/rich_people.json", "rt") as F:
            table_json_source = "".join(F.readlines())

        self.data = SparkContextForTest.spark.createDataFrame([
            [table_json_source, "Who earns 100,000,000? Who has more money? How much money has Donald Trump?"],
            [table_json_source, "How much people earn?"],
            ["  ", "Place holder question"],
            ["  ", " "],
            ["  ", ""],
            ["", " "],
            ["", ""]
        ]).toDF("table_json", "questions")

    def runTest(self):
        document_assembler = MultiDocumentAssembler() \
            .setInputCols("table_json", "questions") \
            .setOutputCols("document_table", "document_questions")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document_questions"]) \
            .setOutputCol("questions")

        table_assembler = TableAssembler()\
            .setInputCols(["document_table"])\
            .setOutputCol("table")

        tapas = TapasForQuestionAnswering() \
            .pretrained()\
            .setMaxSentenceLength(512)\
            .setInputCols(["questions", "table"])\
            .setOutputCol("answers")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            table_assembler,
            tapas
        ])

        model = pipeline.fit(self.data)
        model\
            .transform(self.data)\
            .selectExpr("explode(answers) AS answer")\
            .select("answer")\
            .show(truncate=False)
