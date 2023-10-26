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
class DocumentCharacterTextSplitterTestSpec(unittest.TestCase):
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

        document_character_text_splitter = (
            DocumentCharacterTextSplitter()
            .setInputCols("document")
            .setOutputCol("splits")
            .setChunkSize(20)
            .setChunkOverlap(5)
            .setExplodeSplits(True)
            .setPatternsAreRegex(False)
            .setKeepSeparators(True)
            .setSplitPatterns(["\n\n", "\n", " ", ""])
            .setTrimWhitespace(True)
        )

        pipeline = Pipeline().setStages(
            [document_assembler, document_character_text_splitter]
        )

        pipeline_df = pipeline.fit(df).transform(df)

        results = pipeline_df.select("splits").collect()

        splits = [row["splits"][0].result for row in results]

        expected = [
            "All emotions, and",
            "and that",
            "one particularly,",
            "were abhorrent to",
            "to his cold,",
            "precise but",
            "admirably balanced",
            "mind.",
            "He was, I take it,",
            "it, the most",
            "most perfect",
            "reasoning and",
            "and observing",
            "machine that the",
            "the world has seen.",
        ]

        assert(splits == expected)
