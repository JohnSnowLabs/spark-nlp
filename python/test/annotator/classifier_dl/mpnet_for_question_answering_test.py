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
class MPNetForQuestionAnsweringTestSpec(unittest.TestCase):
    def setUp(self):
        question = (
            "Which name is also used to describe the Amazon rainforest in English?"
        )
        context = (
            "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva "
            "Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: "
            "Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist "
            "broadleaf forest that covers most of the Amazon basin of South America. This basin "
            "encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square "
            "kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes "
            "territory belonging to nine nations. The majority of the forest is contained within "
            "Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and "
            "with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana."
            ' States or departments in four nations contain "Amazonas" in their names. The Amazon'
            " represents over half of the planet's remaining rainforests, and comprises the largest"
            " and most biodiverse tract of tropical rainforest in the world, with an estimated 390"
            " billion individual trees divided into 16,000 species."
        )
        self.data = SparkContextForTest.spark.createDataFrame(
            [[question, context]]
        ).toDF("question", "context")

        self.tested_annotator = (
            MPNetForQuestionAnswering.pretrained()
            .setInputCols("document_question", "document_context")
            .setOutputCol("answer")
            .se
        )

    def test_run(self):
        document_assembler = (
            MultiDocumentAssembler()
            .setInputCols("question", "context")
            .setOutputCols("document_question", "document_context")
        )

        questionAnswering = self.tested_annotator

        pipeline = Pipeline(stages=[document_assembler, questionAnswering])

        model = pipeline.fit(self.data)
        result = model.transform(self.data).select("answer").collect()[0][0][0]
        _, start, end, answer, meta, _ = result
        start = int(meta["start"])
        end = int(meta["end"]) + 1
        score = float(meta["score"])

        expectedStart = 201
        expectedEnd = 230
        expectedAnswer = "Amazonia or the Amazon Jungle"
        expectedScore = 0.09354283660650253

        assert answer == expectedAnswer, "Wrong answer"
        assert start == expectedStart, "Wrong start"
        assert end == expectedEnd, "Wrong end"
        assert round(score, ndigits=3) == round(expectedScore, ndigits=3), "Wrong score"
