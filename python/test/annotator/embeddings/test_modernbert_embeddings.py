"""
Module for testing ModernBertEmbeddings in Python Spark NLP.
"""

import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest


@pytest.mark.slow
class ModernBertEmbeddingsTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def test_embeddings_with_sentence(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        embeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en") \
            .setInputCols("document", "token") \
            .setOutputCol("embeddings") \
            .setMaxSentenceLength(512)

        embeddings_finisher = EmbeddingsFinisher() \
            .setInputCols("embeddings") \
            .setOutputCols("finished_embeddings") \
            .setOutputAsVector(True) \
            .setCleanAnnotations(False)

        pipeline = Pipeline() \
            .setStages([
                document_assembler,
                tokenizer,
                embeddings,
                embeddings_finisher
            ])

        data = self.spark.createDataFrame([
            ["Something is weird on this text"],
            ["ModernBERT is a modern bidirectional encoder model."]
        ]).toDF("text")

        result = pipeline.fit(data).transform(data)
        
        embeddings_result = result.select("finished_embeddings").collect()
        
        # Check that we have embeddings for both sentences
        self.assertEqual(len(embeddings_result), 2)
        
        # Check that embeddings are not empty
        for row in embeddings_result:
            self.assertGreater(len(row.finished_embeddings), 0)
            # Check that each token has embeddings with correct dimension (768 for base model)
            for embedding in row.finished_embeddings:
                self.assertEqual(len(embedding), 768)

    def test_embeddings_with_empty_input(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        embeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en") \
            .setInputCols("document", "token") \
            .setOutputCol("embeddings")

        embeddings_finisher = EmbeddingsFinisher() \
            .setInputCols("embeddings") \
            .setOutputCols("finished_embeddings") \
            .setOutputAsVector(True) \
            .setCleanAnnotations(False)

        pipeline = Pipeline() \
            .setStages([
                document_assembler,
                tokenizer,
                embeddings,
                embeddings_finisher
            ])

        data = self.spark.createDataFrame([[""]]).toDF("text")

        result = pipeline.fit(data).transform(data)
        embeddings_result = result.select("finished_embeddings").collect()
        
        # Should handle empty input gracefully
        self.assertEqual(len(embeddings_result), 1)

    def test_embeddings_with_special_characters(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        embeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en") \
            .setInputCols("document", "token") \
            .setOutputCol("embeddings")

        embeddings_finisher = EmbeddingsFinisher() \
            .setInputCols("embeddings") \
            .setOutputCols("finished_embeddings") \
            .setOutputAsVector(True) \
            .setCleanAnnotations(False)

        pipeline = Pipeline() \
            .setStages([
                document_assembler,
                tokenizer,
                embeddings,
                embeddings_finisher
            ])

        data = self.spark.createDataFrame([
            ["This is a test with @#$%^&*() special characters!"]
        ]).toDF("text")

        result = pipeline.fit(data).transform(data)
        embeddings_result = result.select("finished_embeddings").collect()
        
        # Should handle special characters
        self.assertEqual(len(embeddings_result), 1)
        self.assertGreater(len(embeddings_result[0].finished_embeddings), 0)

    def test_embeddings_batch_processing(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        embeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en") \
            .setInputCols("document", "token") \
            .setOutputCol("embeddings") \
            .setBatchSize(2)

        embeddings_finisher = EmbeddingsFinisher() \
            .setInputCols("embeddings") \
            .setOutputCols("finished_embeddings") \
            .setOutputAsVector(True) \
            .setCleanAnnotations(False)

        pipeline = Pipeline() \
            .setStages([
                document_assembler,
                tokenizer,
                embeddings,
                embeddings_finisher
            ])

        data = self.spark.createDataFrame([
            ["Something is weird on this text"],
            ["ModernBERT is a modern bidirectional encoder model."],
            ["The quick brown fox jumps over the lazy dog."],
            ["I love using Spark NLP for natural language processing tasks."],
            ["Machine learning is revolutionizing how we process text data."]
        ]).toDF("text")

        result = pipeline.fit(data).transform(data)
        embeddings_result = result.select("finished_embeddings").collect()
        
        # Should process all 5 sentences
        self.assertEqual(len(embeddings_result), 5)
        
        # Each sentence should have embeddings
        for row in embeddings_result:
            self.assertGreater(len(row.finished_embeddings), 0)

    def test_parameter_setting(self):
        embeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en") \
            .setMaxSentenceLength(512) \
            .setCaseSensitive(True) \
            .setBatchSize(4) \
            .setDimension(768)

        self.assertEqual(embeddings.getMaxSentenceLength(), 512)
        self.assertTrue(embeddings.getCaseSensitive())
        self.assertEqual(embeddings.getBatchSize(), 4)
        self.assertEqual(embeddings.getDimension(), 768)

    def test_max_sentence_length_validation(self):
        embeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en")
        
        # Test setting valid max sentence length
        embeddings.setMaxSentenceLength(4096)
        self.assertEqual(embeddings.getMaxSentenceLength(), 4096)
        
        # Test that too large max sentence length raises error
        with self.assertRaises(ValueError):
            embeddings.setMaxSentenceLength(8193)
            
        # Test that too small max sentence length raises error
        with self.assertRaises(ValueError):
            embeddings.setMaxSentenceLength(0)

    def test_long_context_support(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        embeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en") \
            .setInputCols("document", "token") \
            .setOutputCol("embeddings") \
            .setMaxSentenceLength(4096)  # Test with longer context

        embeddings_finisher = EmbeddingsFinisher() \
            .setInputCols("embeddings") \
            .setOutputCols("finished_embeddings") \
            .setOutputAsVector(True) \
            .setCleanAnnotations(False)

        pipeline = Pipeline() \
            .setStages([
                document_assembler,
                tokenizer,
                embeddings,
                embeddings_finisher
            ])

        # Create a long text that tests ModernBERT's extended context capabilities
        long_text = "This is a very long sentence that should test the capabilities of ModernBERT with extended context. " * 50
        data = self.spark.createDataFrame([[long_text]]).toDF("text")

        result = pipeline.fit(data).transform(data)
        embeddings_result = result.select("finished_embeddings").collect()
        
        # Should handle long context
        self.assertEqual(len(embeddings_result), 1)
        self.assertGreater(len(embeddings_result[0].finished_embeddings), 0)


if __name__ == "__main__":
    unittest.main()
