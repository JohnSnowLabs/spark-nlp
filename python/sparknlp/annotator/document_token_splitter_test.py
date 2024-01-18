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
from test.util import SparkSessionForTest


@pytest.mark.fast
class DocumentTokenSplitterTestSpec(unittest.TestCase):
    def setUp(self):
        self.data = SparkSessionForTest.spark.createDataFrame(
            [
                [
                    (
                        "All emotions, and that\none particularly, were abhorrent to his cold, precise"
                        " but\nadmirably balanced mind.\n\nHe was, I take it, the most perfect\nreasoning"
                        " and observing machine that the world has seen."
                    )
                ]
            ]
        ).toDF("text")

    def test_run(self):
        df = self.data

        document_assembler = (
            DocumentAssembler().setInputCol("text").setOutputCol("document")
        )

        document_token_splitter = (
            DocumentTokenSplitter()
            .setInputCols("document")
            .setOutputCol("splits")
            .setNumTokens(3)
            .setTokenOverlap(1)
            .setExplodeSplits(True)
            .setTrimWhitespace(True)
        )

        pipeline = Pipeline().setStages([document_assembler, document_token_splitter])

        pipeline_df = pipeline.fit(df).transform(df)

        results = pipeline_df.select("splits").collect()

        splits = [
            row["splits"][0].result.replace("\n\n", " ").replace("\n", " ")
            for row in results
        ]

        expected = [
            "All emotions, and",
            "and that one",
            "one particularly, were",
            "were abhorrent to",
            "to his cold,",
            "cold, precise but",
            "but admirably balanced",
            "balanced mind. He",
            "He was, I",
            "I take it,",
            "it, the most",
            "most perfect reasoning",
            "reasoning and observing",
            "observing machine that",
            "that the world",
            "world has seen.",
        ]

        assert splits == expected
