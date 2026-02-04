/*
 * Copyright 2017-2024 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.ml.ai.VectorDBConnector
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.{AlbertEmbeddings, SentenceEmbeddings, WordEmbeddingsModel}
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec

class VectorDBConnectorTest extends AnyFlatSpec {

  private val spark = SparkSession
    .builder()
    .appName("VectorDBConnectorTest")
    .master("local[*]")
    .config("spark.driver.memory", "8G")
    .config("spark.driver.maxResultSize", "0")
    .config("spark.kryoserializer.buffer.max", "2000M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config(
      "spark.jsl.settings.vectordb.api.key",
      "pcsk_6hPiJc_6jKrVSo45CsYzEXA6q6aJxQjEKHkWLiG1UEqq4LKj1QTK1dwU2EcwLQ8feo99DK" // Set your VectorDB API key here (e.g., Pinecone API key)
    )
    .getOrCreate()

  import spark.implicits._

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  private val embeddings = AlbertEmbeddings
    .pretrained("albert_embeddings_albert_xlarge_v1")
    .setInputCols("document", "token")
    .setOutputCol("word_embeddings")

  private val sentenceEmbeddings = new SentenceEmbeddings()
    .setInputCols("document", "word_embeddings")
    .setOutputCol("sentence_embeddings")
    .setPoolingStrategy("AVERAGE")

  "VectorDBConnector with Pinecone" should "store embeddings with metadata" taggedAs SlowTest in {
    val testData = Seq(
      ("test1", "Apache Spark is a unified analytics engine", "technology"),
      ("test2", "Natural language processing enables AI understanding", "ai"),
      ("test3", "Vector databases provide semantic search", "database"))
      .toDF("id", "text", "category")

    val vectorDB = new VectorDBConnector()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("vectordb_result")
      .setProvider("pinecone")
      .setIndexName("final-index")
      .setNamespace("integration-test")
      .setIdColumn("id")
      .setMetadataColumns(Array("text", "category"))
      .setBatchSize(10)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, sentenceEmbeddings, vectorDB))

    val result = pipeline.fit(testData).transform(testData)
    result.select("vectordb_result").show(false)

    assert(result.count() == 3, "Should process all 3 documents")
  }

  it should "handle large batch sizes correctly" taggedAs SlowTest in {
    val largeData = (1 to 150)
      .map { i =>
        (s"batch_test_$i", s"Test document number $i with content", s"category_${i % 5}")
      }
      .toDF("id", "text", "category")

    val vectorDB = new VectorDBConnector()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("vectordb_result")
      .setProvider("pinecone")
      .setIndexName("final-index")
      .setNamespace("batch-test")
      .setIdColumn("id")
      .setMetadataColumns(Array("category"))
      .setBatchSize(50)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, sentenceEmbeddings, vectorDB))

    val startTime = System.currentTimeMillis()
    val result = pipeline.fit(largeData).transform(largeData)
    val endTime = System.currentTimeMillis()

    assert(result.count() == 150, "Should process all 150 documents")
    println(s"Processed 150 documents in ${(endTime - startTime) / 1000.0} seconds")
  }

  it should "work without metadata columns" taggedAs SlowTest in {
    val testData = Seq(
      ("no_meta_1", "First document without metadata"),
      ("no_meta_2", "Second document without metadata"))
      .toDF("id", "text")

    val vectorDB = new VectorDBConnector()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("vectordb_result")
      .setProvider("pinecone")
      .setIndexName("final-index")
      .setNamespace("no-metadata-test")
      .setIdColumn("id")
      .setBatchSize(10)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, sentenceEmbeddings, vectorDB))

    val result = pipeline.fit(testData).transform(testData)

    assert(result.count() == 2, "Should process both documents without metadata")
  }

  it should "generate UUIDs when no ID column specified" taggedAs SlowTest in {
    val testData = Seq("Document without explicit ID column", "Another document for UUID test")
      .toDF("text")

    val vectorDB = new VectorDBConnector()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("vectordb_result")
      .setProvider("pinecone")
      .setIndexName("final-index")
      .setNamespace("uuid-test")
      .setBatchSize(10)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, sentenceEmbeddings, vectorDB))

    val result = pipeline.fit(testData).transform(testData)

    assert(result.count() == 2, "Should process both documents with auto-generated UUIDs")
  }

  it should "handle different namespaces correctly" taggedAs SlowTest in {
    val testData1 = Seq(("ns1_doc1", "Data for namespace 1")).toDF("id", "text")
    val testData2 = Seq(("ns2_doc1", "Data for namespace 2")).toDF("id", "text")

    val vectorDB1 = new VectorDBConnector()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("vectordb_result")
      .setProvider("pinecone")
      .setIndexName("final-index")
      .setNamespace("namespace-1")
      .setIdColumn("id")
      .setBatchSize(10)

    val vectorDB2 = new VectorDBConnector()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("vectordb_result")
      .setProvider("pinecone")
      .setIndexName("final-index")
      .setNamespace("namespace-2")
      .setIdColumn("id")
      .setBatchSize(10)

    val pipeline1 = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, sentenceEmbeddings, vectorDB1))

    val pipeline2 = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, sentenceEmbeddings, vectorDB2))

    val result1 = pipeline1.fit(testData1).transform(testData1)
    val result2 = pipeline2.fit(testData2).transform(testData2)

    assert(result1.count() == 1, "Should store in namespace 1")
    assert(result2.count() == 1, "Should store in namespace 2")
  }

  it should "properly escape special characters in metadata" taggedAs SlowTest in {
    val testData = Seq(
      ("escape_test_1", "Text with \"quotes\"", "category with spaces"),
      ("escape_test_2", "Text with\nnewlines\tand\ttabs", "special/chars\\here"))
      .toDF("id", "text", "category")

    val vectorDB = new VectorDBConnector()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("vectordb_result")
      .setProvider("pinecone")
      .setIndexName("final-index")
      .setNamespace("escape-test")
      .setIdColumn("id")
      .setMetadataColumns(Array("text", "category"))
      .setBatchSize(10)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, sentenceEmbeddings, vectorDB))

    val result = pipeline.fit(testData).transform(testData)

    assert(result.count() == 2, "Should handle special characters in metadata")
  }

  it should "process multiple metadata columns correctly" taggedAs SlowTest in {
    val testData = Seq(
      ("multi_meta_1", "Document one", "tech", "2024-01-15", "author1", "published"),
      ("multi_meta_2", "Document two", "ai", "2024-01-16", "author2", "draft"))
      .toDF("id", "text", "category", "date", "author", "status")

    val vectorDB = new VectorDBConnector()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("vectordb_result")
      .setProvider("pinecone")
      .setIndexName("final-index")
      .setNamespace("multi-metadata-test")
      .setIdColumn("id")
      .setMetadataColumns(Array("text", "category", "date", "author", "status"))
      .setBatchSize(10)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, embeddings, sentenceEmbeddings, vectorDB))

    val result = pipeline.fit(testData).transform(testData)

    assert(result.count() == 2, "Should process documents with multiple metadata fields")
  }
}
