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
from pyspark.sql.types import StructType, StructField
from pyspark.sql import Row

from sparknlp.base import GGUFRankingFinisher
from sparknlp.annotation import Annotation
from test.util import SparkContextForTest


@pytest.mark.fast
class GGUFRankingFinisherTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def create_mock_reranker_output(self):
        """Create mock data to simulate AutoGGUFReranker output."""

        documents = [
            ("A man is eating food.", 0.85, "A man is eating pasta."),
            ("A man is eating a piece of bread.", 0.72, "A man is eating pasta."),
            ("The girl is carrying a baby.", 0.15, "A man is eating pasta."),
            ("A man is riding a horse.", 0.28, "A man is eating pasta."),
            ("A young girl is playing violin.", 0.05, "A man is eating pasta."),
        ]

        annotations = []
        for text, score, query in documents:
            annotation = Annotation(
                annotatorType="document",
                begin=0,
                end=len(text) - 1,
                result=text,
                metadata={"relevance_score": str(score), "query": query},
                embeddings=[],
            )
            annotations.append(annotation)

        # Create DataFrame with annotation array
        rows = [Row(reranked_documents=annotations)]
        schema = StructType(
            [StructField("reranked_documents", Annotation.arrayType(), nullable=False)]
        )

        return self.spark.createDataFrame(rows, schema)

    def test_default_settings(self):
        """Test GGUFRankingFinisher with default settings."""
        mock_data = self.create_mock_reranker_output()

        finisher = (
            GGUFRankingFinisher()
            .setInputCols("reranked_documents")
            .setOutputCol("ranked_documents")
        )

        result = finisher.transform(mock_data)

        self.assertIn("ranked_documents", result.columns)

        # Get the ranked documents
        ranked_docs = result.collect()[0]["ranked_documents"]

        self.assertEqual(len(ranked_docs), 5)

        # Check that results are sorted by relevance score in descending order
        scores = [float(doc.metadata["relevance_score"]) for doc in ranked_docs]
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Check that rank metadata is added
        ranks = [int(doc.metadata["rank"]) for doc in ranked_docs]
        self.assertEqual(ranks, [1, 2, 3, 4, 5])

    def test_top_k(self):
        """Test GGUFRankingFinisher with topK setting."""
        mock_data = self.create_mock_reranker_output()

        finisher = (
            GGUFRankingFinisher()
            .setInputCols("reranked_documents")
            .setOutputCol("ranked_documents")
            .setTopK(3)
        )

        result = finisher.transform(mock_data)

        ranked_docs = result.collect()[0]["ranked_documents"]
        self.assertEqual(len(ranked_docs), 3)

        # Check that we get the top 3 scores
        scores = [float(doc.metadata["relevance_score"]) for doc in ranked_docs]
        self.assertEqual(len(scores), 3)
        self.assertIn(0.85, scores)
        self.assertIn(0.72, scores)
        self.assertIn(0.28, scores)

        # Check ranks are correct
        ranks = [int(doc.metadata["rank"]) for doc in ranked_docs]
        self.assertEqual(ranks, [1, 2, 3])

    def test_threshold_filtering(self):
        """Test GGUFRankingFinisher with minimum relevance score threshold."""
        mock_data = self.create_mock_reranker_output()

        finisher = (
            GGUFRankingFinisher()
            .setInputCols("reranked_documents")
            .setOutputCol("ranked_documents")
            .setMinRelevanceScore(0.3)
        )

        result = finisher.transform(mock_data)

        ranked_docs = result.collect()[0]["ranked_documents"]
        self.assertEqual(len(ranked_docs), 2)  # Only scores >= 0.3 (0.85 and 0.72)

        scores = [float(doc.metadata["relevance_score"]) for doc in ranked_docs]
        self.assertTrue(all(score >= 0.3 for score in scores))

    def test_min_max_scaling(self):
        """Test GGUFRankingFinisher with min-max scaling."""
        mock_data = self.create_mock_reranker_output()

        finisher = (
            GGUFRankingFinisher()
            .setInputCols("reranked_documents")
            .setOutputCol("ranked_documents")
            .setMinMaxScaling(True)
        )

        result = finisher.transform(mock_data)

        ranked_docs = result.collect()[0]["ranked_documents"]
        scores = [float(doc.metadata["relevance_score"]) for doc in ranked_docs]

        # Check that scores are between 0 and 1
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))

        # Check that we have both min (0.0) and max (1.0) values
        self.assertIn(1.0, scores)  # Max original score should be 1.0
        self.assertIn(0.0, scores)  # Min original score should be 0.0

    def test_combined_filters(self):
        """Test GGUFRankingFinisher with combined topK, threshold, and scaling."""
        mock_data = self.create_mock_reranker_output()

        finisher = (
            GGUFRankingFinisher()
            .setInputCols("reranked_documents")
            .setOutputCol("ranked_documents")
            .setTopK(2)
            .setMinRelevanceScore(0.1)
            .setMinMaxScaling(True)
        )

        result = finisher.transform(mock_data)

        ranked_docs = result.collect()[0]["ranked_documents"]

        # Should have at most 2 results due to topK
        self.assertLessEqual(len(ranked_docs), 2)

        scores = [float(doc.metadata["relevance_score"]) for doc in ranked_docs]

        # All scores should be >= 0.1 and <= 1.0
        self.assertTrue(all(0.1 <= score <= 1.0 for score in scores))

        # Results should be sorted descending
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Check that ranks are correct
        ranks = [int(doc.metadata["rank"]) for doc in ranked_docs]
        self.assertEqual(ranks, list(range(1, len(ranked_docs) + 1)))

    def test_empty_input(self):
        """Test GGUFRankingFinisher with empty input."""

        # Create empty annotations
        rows = [Row(reranked_documents=[])]
        schema = StructType(
            [StructField("reranked_documents", Annotation.arrayType(), nullable=False)]
        )

        empty_data = self.spark.createDataFrame(rows, schema)

        finisher = (
            GGUFRankingFinisher()
            .setInputCols("reranked_documents")
            .setOutputCol("ranked_documents")
        )

        result = finisher.transform(empty_data)

        # Since empty rows are filtered out, the result should have no rows
        result_count = result.count()
        self.assertEqual(result_count, 0)

    def test_query_preservation(self):
        """Test that query information is preserved in metadata."""
        mock_data = self.create_mock_reranker_output()

        finisher = (
            GGUFRankingFinisher()
            .setInputCols("reranked_documents")
            .setOutputCol("ranked_documents")
        )

        result = finisher.transform(mock_data)

        ranked_docs = result.collect()[0]["ranked_documents"]

        # Check that query information is preserved in metadata
        for doc in ranked_docs:
            self.assertIn("query", doc.metadata)
            self.assertEqual(doc.metadata["query"], "A man is eating pasta.")

    def test_missing_relevance_scores(self):
        """Test handling of documents with missing relevance scores."""

        documents = [
            (
                "A man is eating food.",
                {"relevance_score": "0.85", "query": "A man is eating pasta."},
            ),
            (
                "A man is eating a piece of bread.",
                {"query": "A man is eating pasta."},
            ),  # Missing score
            (
                "The girl is carrying a baby.",
                {"relevance_score": "0.15", "query": "A man is eating pasta."},
            ),
        ]

        annotations = []
        for text, metadata in documents:
            annotation = Annotation(
                annotatorType="document",
                begin=0,
                end=len(text) - 1,
                result=text,
                metadata=metadata,
                embeddings=[],
            )
            annotations.append(annotation)

        rows = [Row(reranked_documents=annotations)]
        schema = StructType(
            [StructField("reranked_documents", Annotation.arrayType(), nullable=False)]
        )

        test_data = self.spark.createDataFrame(rows, schema)

        finisher = (
            GGUFRankingFinisher()
            .setInputCols("reranked_documents")
            .setOutputCol("ranked_documents")
        )

        result = finisher.transform(test_data)

        ranked_docs = result.collect()[0]["ranked_documents"]
        self.assertEqual(len(ranked_docs), 3)

        # Document with missing score should get 0.0 and be ranked last
        scores = [float(doc.metadata["relevance_score"]) for doc in ranked_docs]
        self.assertEqual(scores[-1], 0.0)  # Missing score becomes 0.0

    def test_parameter_getters_setters(self):
        """Test parameter getters and setters."""
        finisher = GGUFRankingFinisher()

        # Test topK
        finisher.setTopK(5)
        self.assertEqual(finisher.getTopK(), 5)

        # Test minRelevanceScore
        finisher.setMinRelevanceScore(0.5)
        self.assertEqual(finisher.getMinRelevanceScore(), 0.5)

        # Test minMaxScaling
        finisher.setMinMaxScaling(True)
        self.assertTrue(finisher.getMinMaxScaling())

        # Test outputCol
        finisher.setOutputCol("custom_output")
        self.assertEqual(finisher.getOutputCol(), "custom_output")


if __name__ == "__main__":
    unittest.main()
