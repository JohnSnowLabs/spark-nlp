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
import os
import shutil
import tempfile
import unittest

import pytest
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, explode

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkSessionForTest


@pytest.mark.fast
class BM25TestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSessionForTest.spark
        self.data = self.spark.createDataFrame([
            (1, "Apples are a great source of dietary fiber and vitamin C."),
            (2, "Bananas are rich in potassium and natural sugars."),
            (3, "Machine learning is a subset of artificial intelligence."),
            (4, "Deep learning uses neural networks with many layers."),
            (5, "Florence in Italy is one of the most beautiful cities in Europe."),
            (6, "The French Riviera is a warm coastal region in southern France."),
            (7, "Vitamin C deficiency can lead to scurvy and immune system problems."),
            (8, "Neural networks are inspired by the human brain's structure."),
            (9, "Potassium is an essential mineral for heart and muscle function."),
            (10, "Italy is home to some of the world's greatest Renaissance art."),
        ], ["id", "text"])

        self.document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")
        self.tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")
        self.stop_words_cleaner = StopWordsCleaner() \
            .setInputCols(["token"]) \
            .setOutputCol("clean_token") \
            .setCaseSensitive(False)

    def _fit(self):
        bm25 = BM25Approach() \
            .setInputCols(["clean_token"]) \
            .setOutputCol("bm25_rankings") \
            .setK1(1.2) \
            .setB(0.75) \
            .setMinDocFreq(1) \
            .setCaseSensitive(False)
        pipeline = Pipeline(stages=[
            self.document_assembler, self.tokenizer, self.stop_words_cleaner, bm25])
        return pipeline.fit(self.data)

    def _ranked(self, model):
        result = model.transform(self.data)
        return (
            result
            .select(col("id"), explode(col("bm25_rankings")).alias("ranking"))
            .select(
                col("id"),
                col("ranking.metadata")["bm25_score"].cast("double").alias("bm25_score"),
                col("ranking.metadata")["num_query_terms_matched"].cast("int").alias("terms_matched"))
            .orderBy(col("bm25_score").desc())
            .collect()
        )

    def runTest(self):
        pipeline_model = self._fit()
        bm25_model = pipeline_model.stages[-1]

        # 1. Statistics learned during fit
        self.assertEqual(bm25_model.getNumDocuments(), 10)
        self.assertTrue(bm25_model.getAvgDocLength() > 0.0)

        # 2. Ranking with a global query
        bm25_model.setQuery("vitamin C health benefits fruits")
        ranked = self._ranked(pipeline_model)
        for row in ranked:
            print(f"id={row['id']} score={row['bm25_score']} matched={row['terms_matched']}")
        top_two = {ranked[0]["id"], ranked[1]["id"]}
        self.assertEqual(top_two, {1, 7})
        self.assertTrue(all(r["bm25_score"] >= 0.0 for r in ranked))

        # 3. Reuse the same fitted model with a different query
        bm25_model.setQuery("neural networks deep learning")
        ai_ranked = self._ranked(pipeline_model)
        self.assertIn(ai_ranked[0]["id"], {4, 8})

        # 4. Save and load the fitted model, then reuse it
        tmp_dir = tempfile.mkdtemp()
        model_path = os.path.join(tmp_dir, "bm25_corpus_model")
        try:
            bm25_model.write().overwrite().save(model_path)
            loaded = BM25Model.load(model_path)
            self.assertEqual(loaded.getNumDocuments(), 10)

            loaded.setQuery("Italy France Europe travel")
            infer_pipeline = Pipeline(stages=[
                self.document_assembler, self.tokenizer, self.stop_words_cleaner, loaded])
            infer_model = infer_pipeline.fit(self.data)
            travel_ranked = self._ranked(infer_model)
            self.assertIn(travel_ranked[0]["id"], {5, 6, 10})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # 5. Non-default tuning params must propagate from the approach to the fitted model, and
        # minDocFreq must prune rare terms. "vitamin" appears in only 2 documents, so minDocFreq=3
        # drops it; a query made solely of pruned terms then scores 0 for every document.
        custom = (
            BM25Approach()
            .setInputCols(["clean_token"])
            .setOutputCol("bm25_rankings")
            .setK1(2.0)
            .setB(0.5)
            .setMinDocFreq(3)
        )
        custom_pipeline_model = Pipeline(stages=[
            self.document_assembler, self.tokenizer, self.stop_words_cleaner, custom]).fit(self.data)
        custom_model = custom_pipeline_model.stages[-1]
        self.assertEqual(custom_model.getK1(), 2.0)
        self.assertEqual(custom_model.getB(), 0.5)

        custom_model.setQuery("vitamin")
        self.assertTrue(all(r["bm25_score"] == 0.0 for r in self._ranked(custom_pipeline_model)))
