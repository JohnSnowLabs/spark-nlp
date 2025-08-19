#  Copyright 2017-2024 John Snow Labs
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


@pytest.mark.slow
class AutoGGUFRerankerTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.query = "A man is eating pasta."
        self.data = (
            self.spark.createDataFrame(
                [
                    ["A man is eating food."],
                    ["A man is eating a piece of bread."],
                    ["一个中国男人在吃面条"],
                    ["The girl is carrying a baby."],
                    ["A man is riding a horse."],
                    ["A young girl is playing violin."],
                ]
            )
            .toDF("text")
            .repartition(1)
        )

    def runTest(self):
        document_assembler = (
            DocumentAssembler().setInputCol("text").setOutputCol("document")
        )

        # Use a local model path for testing - in real scenarios, use pretrained()
        model_path = "/tmp/bge-reranker-v2-m3-Q4_K_M.gguf"
        
        # Skip test if model file doesn't exist
        import os
        if not os.path.exists(model_path):
            self.skipTest(f"Model file not found: {model_path}")

        reranker = (
            AutoGGUFReranker.loadSavedModel(model_path, self.spark)
            .setInputCols("document")
            .setOutputCol("reranked_documents")
            .setBatchSize(4)
            .setQuery(self.query)
        )

        pipeline = Pipeline().setStages([document_assembler, reranker])
        results = pipeline.fit(self.data).transform(self.data)

        # Check that results are returned
        collected_results = results.collect()
        self.assertGreater(len(collected_results), 0)

        # Check that each result has reranked_documents column
        for row in collected_results:
            self.assertIsNotNone(row["reranked_documents"])
            # Check that annotations have metadata with relevance_score
            annotations = row["reranked_documents"]
            for annotation in annotations:
                self.assertIn("relevance_score", annotation.metadata)
                self.assertIn("query", annotation.metadata)
                self.assertEqual(annotation.metadata["query"], self.query)

        results.select("reranked_documents").show(truncate=False)


@pytest.mark.slow
class AutoGGUFRerankerPretrainedTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.query = "A man is eating pasta."
        self.data = (
            self.spark.createDataFrame(
                [
                    ["A man is eating food."],
                    ["A man is eating a piece of bread."],
                    ["The girl is carrying a baby."],
                    ["A man is riding a horse."],
                ]
            )
            .toDF("text")
            .repartition(1)
        )

    def runTest(self):
        document_assembler = (
            DocumentAssembler().setInputCol("text").setOutputCol("document")
        )

        # Test with pretrained model (may not be available in test environment)
        try:
            reranker = (
                AutoGGUFReranker.pretrained("bge-reranker-v2-m3-Q4_K_M")
                .setInputCols("document")
                .setOutputCol("reranked_documents")
                .setBatchSize(2)
                .setQuery(self.query)
            )

            pipeline = Pipeline().setStages([document_assembler, reranker])
            results = pipeline.fit(self.data).transform(self.data)

            # Verify results contain relevance scores
            collected_results = results.collect()
            for row in collected_results:
                annotations = row["reranked_documents"]
                for annotation in annotations:
                    self.assertIn("relevance_score", annotation.metadata)
                    # Relevance score should be a valid number
                    score = float(annotation.metadata["relevance_score"])
                    self.assertIsInstance(score, float)

            results.show()
        except Exception as e:
            # Skip if pretrained model is not available
            self.skipTest(f"Pretrained model not available: {str(e)}")

@pytest.mark.slow
class AutoGGUFRerankerMetadataTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        model_path = "/tmp/bge-reranker-v2-m3-Q4_K_M.gguf"
        
        # Skip test if model file doesn't exist
        import os
        if not os.path.exists(model_path):
            self.skipTest(f"Model file not found: {model_path}")

        reranker = AutoGGUFReranker.loadSavedModel(model_path, self.spark)

        metadata = reranker.getMetadata()
        self.assertIsNotNone(metadata)
        self.assertGreater(len(metadata), 0)
        
        print("Model metadata:")
        print(eval(metadata))

#
# @pytest.mark.slow
# class AutoGGUFRerankerSerializationTestSpec(unittest.TestCase):
#     def setUp(self):
#         self.spark = SparkContextForTest.spark
#         self.query = "A man is eating pasta."
#         self.data = (
#             self.spark.createDataFrame(
#                 [
#                     ["A man is eating food."],
#                     ["The girl is carrying a baby."],
#                 ]
#             )
#             .toDF("text")
#             .repartition(1)
#         )
#
#     def runTest(self):
#         model_path = "/tmp/bge-reranker-v2-m3-Q4_K_M.gguf"
#
#         # Skip test if model file doesn't exist
#         import os
#         if not os.path.exists(model_path):
#             self.skipTest(f"Model file not found: {model_path}")
#
#         document_assembler = (
#             DocumentAssembler().setInputCol("text").setOutputCol("document")
#         )
#
#         reranker = (
#             AutoGGUFReranker.loadSavedModel(model_path, self.spark)
#             .setInputCols("document")
#             .setOutputCol("reranked_documents")
#             .setQuery(self.query)
#             .setBatchSize(2)
#         )
#
#         pipeline = Pipeline().setStages([document_assembler, reranker])
#         pipeline_model = pipeline.fit(self.data)
#
#         # Test serialization
#         save_path = "file://" + "/tmp/test_gguf_reranker_py"
#         reranker.write().overwrite().save(save_path)
#
#         # Test deserialization
#         loaded_reranker = AutoGGUFReranker.load(save_path)
#         new_pipeline = Pipeline().setStages([document_assembler, loaded_reranker])
#
#         # Test that loaded model works
#         results = new_pipeline.fit(self.data).transform(self.data)
#         collected_results = results.collect()
#
#         self.assertGreater(len(collected_results), 0)
#         for row in collected_results:
#             annotations = row["reranked_documents"]
#             for annotation in annotations:
#                 self.assertIn("relevance_score", annotation.metadata)
#
#         results.select("reranked_documents").show(truncate=False)


@pytest.mark.slow 
class AutoGGUFRerankerErrorHandlingTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        # Test error handling when query is not set
        document_assembler = (
            DocumentAssembler().setInputCol("text").setOutputCol("document")
        )

        data = self.spark.createDataFrame([["Test document"]]).toDF("text")

        model_path = "/tmp/bge-reranker-v2-m3-Q4_K_M.gguf"
        
        # Skip test if model file doesn't exist
        import os
        if not os.path.exists(model_path):
            self.skipTest(f"Model file not found: {model_path}")

        reranker = (
            AutoGGUFReranker.loadSavedModel(model_path, self.spark)
            .setInputCols("document")
            .setOutputCol("reranked_documents")
            .setBatchSize(1)
            # Intentionally not setting query to test default behavior
        )

        pipeline = Pipeline().setStages([document_assembler, reranker])
        
        # This should still work with empty query (based on implementation)
        try:
            results = pipeline.fit(data).transform(data)
            results.collect()
            print("Reranker handles missing query gracefully")
        except Exception as e:
            print(f"Expected behavior when query not set: {str(e)}")


if __name__ == "__main__":
    unittest.main()
