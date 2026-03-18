#  Copyright 2017-2026 John Snow Labs
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
from pyspark.ml import Pipeline

from sparknlp.annotator import DocumentTitleSplitter
from sparknlp.reader.reader2doc import Reader2Doc
from test.util import SparkContextForTest


@pytest.mark.fast
class DocumentTitleSplitterTestSpec(unittest.TestCase):
    def setUp(self):
        spark = SparkContextForTest.spark
        self.empty_df = spark.createDataFrame([], "string").toDF("text")

    def test_run(self):
        reader2doc = (
            Reader2Doc()
            .setContentType("text/markdown")
            .setContentPath(
                f"file:///{os.getcwd()}/../src/test/resources/reader/md/title-chunking.md"
            )
            .setOutputCol("document")
            .setOutputAsDocument(False)
            .setExplodeDocs(False)
        )

        document_title_splitter = (
            DocumentTitleSplitter()
            .setInputCols(["document"])
            .setOutputCol("splits")
            .setExplodeSplits(True)
        )

        pipeline = Pipeline().setStages([reader2doc, document_title_splitter])
        pipeline_df = pipeline.fit(self.empty_df).transform(self.empty_df)

        results = pipeline_df.select("splits").collect()
        splits = [row["splits"][0].result for row in results]

        assert len(splits) == 3
        assert splits[0].startswith("Overview")
        assert splits[1].startswith("Configuration")
        assert splits[2].startswith("Example")
