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
import unittest

import pytest

from sparknlp.base import *
from pyspark.ml import Pipeline
from test.util import SparkSessionForTest


@pytest.mark.fast
class AnnotationMergerTest(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSessionForTest.spark

    def test_merge_two_document_columns(self):
        data = self.spark.createDataFrame(
            [("Hello world", "This is a table")], ["text1", "text2"]
        )

        da1 = DocumentAssembler().setInputCol("text1").setOutputCol("document_text")
        da2 = DocumentAssembler().setInputCol("text2").setOutputCol("document_table")

        merger = (
            AnnotationMerger()
            .setInputCols(["document_text", "document_table"])
            .setOutputCol("merged")
        )

        pipeline = Pipeline().setStages([da1, da2, merger]).fit(data)
        result = pipeline.transform(data)

        merged = result.select("merged").collect()[0]["merged"]
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["result"], "Hello world")
        self.assertEqual(merged[1]["result"], "This is a table")

    def test_merge_three_columns(self):
        data = self.spark.createDataFrame(
            [("First", "Second", "Third")], ["text1", "text2", "text3"]
        )

        da1 = DocumentAssembler().setInputCol("text1").setOutputCol("doc1")
        da2 = DocumentAssembler().setInputCol("text2").setOutputCol("doc2")
        da3 = DocumentAssembler().setInputCol("text3").setOutputCol("doc3")

        merger = (
            AnnotationMerger()
            .setInputCols(["doc1", "doc2", "doc3"])
            .setOutputCol("merged")
        )

        pipeline = Pipeline().setStages([da1, da2, da3, merger]).fit(data)
        result = pipeline.transform(data)

        merged = result.select("merged").collect()[0]["merged"]
        self.assertEqual(len(merged), 3)
        results = [a["result"] for a in merged]
        self.assertEqual(results, ["First", "Second", "Third"])

    def test_single_column_passthrough(self):
        data = self.spark.createDataFrame([("Hello world",)], ["text"])

        da = DocumentAssembler().setInputCol("text").setOutputCol("document")

        merger = (
            AnnotationMerger()
            .setInputCols(["document"])
            .setOutputCol("merged")
        )

        pipeline = Pipeline().setStages([da, merger]).fit(data)
        result = pipeline.transform(data)

        merged = result.select("merged").collect()[0]["merged"]
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["result"], "Hello world")

    def test_sort_by_begin(self):
        data = self.spark.createDataFrame(
            [("Second part", "First")], ["text1", "text2"]
        )

        da1 = DocumentAssembler().setInputCol("text1").setOutputCol("doc1")
        da2 = DocumentAssembler().setInputCol("text2").setOutputCol("doc2")

        merger = (
            AnnotationMerger()
            .setInputCols(["doc1", "doc2"])
            .setOutputCol("merged")
            .setSortByBegin(True)
        )

        pipeline = Pipeline().setStages([da1, da2, merger]).fit(data)
        result = pipeline.transform(data)

        merged = result.select("merged").collect()[0]["merged"]
        self.assertEqual(len(merged), 2)
        begins = [a["begin"] for a in merged]
        self.assertEqual(begins, sorted(begins))

    def test_output_annotator_type(self):
        data = self.spark.createDataFrame(
            [("Hello", "World")], ["text1", "text2"]
        )

        da1 = DocumentAssembler().setInputCol("text1").setOutputCol("doc1")
        da2 = DocumentAssembler().setInputCol("text2").setOutputCol("doc2")

        merger = (
            AnnotationMerger()
            .setInputCols(["doc1", "doc2"])
            .setOutputCol("merged")
            .setOutputAsAnnotatorType("chunk")
        )

        pipeline = Pipeline().setStages([da1, da2, merger]).fit(data)
        result = pipeline.transform(data)

        merged = result.select("merged").collect()[0]["merged"]
        for annotation in merged:
            self.assertEqual(annotation["annotatorType"], "chunk")

    def test_multiple_rows(self):
        data = self.spark.createDataFrame(
            [
                ("Row 1 A", "Row 1 B"),
                ("Row 2 A", "Row 2 B"),
                ("Row 3 A", "Row 3 B"),
            ],
            ["text1", "text2"],
        )

        da1 = DocumentAssembler().setInputCol("text1").setOutputCol("doc1")
        da2 = DocumentAssembler().setInputCol("text2").setOutputCol("doc2")

        merger = (
            AnnotationMerger()
            .setInputCols(["doc1", "doc2"])
            .setOutputCol("merged")
        )

        pipeline = Pipeline().setStages([da1, da2, merger]).fit(data)
        result = pipeline.transform(data)

        rows = result.select("merged").collect()
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(len(row["merged"]), 2)

    def test_pipeline_serialization(self):
        import tempfile
        import os

        data = self.spark.createDataFrame(
            [("Hello", "World")], ["text1", "text2"]
        )

        da1 = DocumentAssembler().setInputCol("text1").setOutputCol("doc1")
        da2 = DocumentAssembler().setInputCol("text2").setOutputCol("doc2")

        merger = (
            AnnotationMerger()
            .setInputCols(["doc1", "doc2"])
            .setOutputCol("merged")
        )

        pipeline = Pipeline().setStages([da1, da2, merger]).fit(data)

        tmp_path = os.path.join(tempfile.mkdtemp(), "annotation_merger_test")
        pipeline.write().overwrite().save(tmp_path)

        from pyspark.ml import PipelineModel

        loaded = PipelineModel.load(tmp_path)
        self.assertIsNotNone(loaded)

        result = loaded.transform(data)
        merged = result.select("merged").collect()[0]["merged"]
        self.assertEqual(len(merged), 2)

    def test_reader_assembler_merge_document_text_and_table(self):
        import os
        from sparknlp.reader.reader_assembler import ReaderAssembler

        empty_df = self.spark.createDataFrame([], "string").toDF("text")

        reader = (
            ReaderAssembler()
            .setContentType("application/msword")
            .setContentPath(f"file:///{os.getcwd()}/../src/test/resources/reader/doc/doc-img-table.docx")
            .setOutputCol("document")
        )

        merger = (
            AnnotationMerger()
            .setInputCols(["document_text", "document_table"])
            .setOutputCol("merged")
        )

        pipeline = Pipeline(stages=[reader, merger])
        model = pipeline.fit(empty_df)
        result = model.transform(empty_df)

        self.assertTrue(result.count() > 0)

        merged = result.select("merged").collect()[0]["merged"]
        self.assertGreaterEqual(len(merged), 2)

        for annotation in merged:
            self.assertEqual(annotation["annotatorType"], "document")


if __name__ == "__main__":
    unittest.main()

