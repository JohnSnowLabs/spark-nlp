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
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession


@pytest.mark.fast
class VectorDBConnectorTestSpec(unittest.TestCase):
    """Test VectorDBConnector with Pinecone integration using metadata columns and large batch."""
    def setUp(self):

        self.spark = SparkSession.builder \
            .appName("VectorDBConnectorTest") \
            .master("local[*]") \
            .config("spark.driver.memory", "8G") \
            .config("spark.driver.maxResultSize", "0") \
            .config("spark.kryoserializer.buffer.max", "2000M") \
            .config("spark.jars", "lib/sparknlp.jar") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.jsl.settings.vectordb.api.key",
                    "pcsk_5hgJrG_9dcAfb45diMzapbTSvorEDSEdUDA9gtexG7ywuAr7Ahrf2WnF2bZiFNRGP5RmRq") \
            .getOrCreate()

        self.document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        self.tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        self.embeddings = AlbertEmbeddings.pretrained("albert_embeddings_albert_xlarge_v1") \
            .setInputCols(["document", "token"]) \
            .setOutputCol("word_embeddings")

        self.sentence_embeddings = SentenceEmbeddings() \
            .setInputCols(["document", "word_embeddings"]) \
            .setOutputCol("sentence_embeddings") \
            .setPoolingStrategy("AVERAGE")


    def test_run(self):
        # Create large dataset with multiple metadata columns
        large_data = self.spark.createDataFrame([
            (f"test_id_{i}",
             f"Test document number {i} with content for vector database integration",
             f"category_{i % 5}",
             f"2024-01-{(i % 28) + 1:02d}",
             f"author_{i % 10}",
             "published" if i % 2 == 0 else "draft")
            for i in range(1, 201)
        ]).toDF("id", "text", "category", "date", "author", "status")

        vector_db = VectorDBConnector() \
            .setInputCols(["document", "sentence_embeddings"]) \
            .setOutputCol("vectordb_result") \
            .setProvider("pinecone") \
            .setIndexName("this") \
            .setNamespace("python-integration-test") \
            .setIdColumn("id") \
            .setMetadataColumns(["text", "category", "date", "author", "status"]) \
            .setBatchSize(50)

        pipeline = Pipeline(stages=[
            self.document_assembler,
            self.tokenizer,
            self.embeddings,
            self.sentence_embeddings,
            vector_db
        ])

        result = pipeline.fit(large_data).transform(large_data)
        result.select("vectordb_result").show(truncate=False)

        self.assertEqual(result.count(), 200, "Should process all 200 documents with metadata")
