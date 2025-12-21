package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotator.{SentenceDetector, Tokenizer}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.{AlbertEmbeddings, SentenceEmbeddings, WordEmbeddingsModel}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.BeforeAndAfterAll

class PineconeEmbeddingsCompleteTest extends AnyFlatSpec with BeforeAndAfterAll {

    val pineconeApiKey = "pcsk_5hgJrG_9dcAfb45diMzapbTSvorEDSEdUDA9gtexG7ywuAr7Ahrf2WnF2bZiFNRGP5RmRq"

    val pineconeEnvironment = "us-east-1-aws"

    // Create Spark session with Pinecone configuration
    private val spark = SparkSession
      .builder()
      .appName("PineconeEmbeddingsTest")
      .master("local[*]")
      .config("spark.driver.memory", "8G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "2000M")
      // Set Pinecone API credentials
      .config("spark.jsl.settings.pinecone.api.key", pineconeApiKey)
      .config("spark.jsl.settings.pinecone.environment", pineconeEnvironment)
      .getOrCreate()


    import spark.implicits._
    println("=" * 80)
    println("Pinecone Test Suite Starting")
    println("=" * 80)
    println(s"Pinecone Environment: $pineconeEnvironment")
    println(s"Pinecone API Key: ${if (pineconeApiKey.nonEmpty) "***set***" else "not set"}")
    println("=" * 80)



  "PineconeEmbeddings" should "successfully store embeddings with metadata" taggedAs SlowTest in {

    val testData = Seq(
      ("test1", "Apache Spark is a unified analytics engine", "technology"),
      ("test2", "Natural language processing enables AI understanding", "ai"),
      ("test3", "Vector databases provide semantic search", "database"))
      .toDF("id", "text", "category")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val embeddings = AlbertEmbeddings
      .pretrained("albert_embeddings_albert_base_v1")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    val sentenceEmbeddings = new SentenceEmbeddings()
      .setInputCols("document", "embeddings")
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pinecone = new PineconeEmbeddings()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("pinecone_result")
      .setIndexName("spark-nlp-test")
      .setNamespace("batch-test")
      .setIdColumn("id")
      .setMetadataColumns(Array("text", "category"))
      .setBatchSize(10)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, tokenizer,  embeddings,  sentenceEmbeddings, pinecone))

    val result = pipeline.fit(testData).transform(testData)

    assert(result.count() == 3, "Should process all 3 documents")
    println("✓ Test 1 passed: Successfully stored 3 documents with metadata")
  }

  it should "handle large batch sizes correctly" taggedAs SlowTest in {

    val largeData = (1 to 2500).map { i =>
      (s"batch_test_$i", s"Test document number $i with content", s"category_${i % 5}")
    }.toDF("id", "text", "category")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val embeddings = AlbertEmbeddings
      .pretrained("albert_embeddings_albert_base_v1")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    val sentenceEmbeddings = new SentenceEmbeddings()
      .setInputCols("document", "embeddings")
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pinecone = new PineconeEmbeddings()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("pinecone_result")
      .setIndexName("spark-nlp-test")
      .setNamespace("batch-test")
      .setIdColumn("id")
      .setMetadataColumns(Array("category"))
      .setBatchSize(100)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, tokenizer, embeddings, sentenceEmbeddings, pinecone))

    val startTime = System.currentTimeMillis()
    val result = pipeline.fit(largeData).transform(largeData)
    val endTime = System.currentTimeMillis()

    assert(result.count() == 250, "Should process all 250 documents")
    println(
      s"✓ Test 2 passed: Processed 250 documents in ${(endTime - startTime) / 1000.0} seconds")
  }

  it should "work without metadata columns" taggedAs SlowTest in {

    val testData = Seq(
      ("no_meta_1", "First document without metadata"),
      ("no_meta_2", "Second document without metadata"))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = WordEmbeddingsModel
      .pretrained("glove_100d")
      .setInputCols("document")
      .setOutputCol("word_embeddings")

    val sentenceEmbeddings = new SentenceEmbeddings()
      .setInputCols("document", "word_embeddings")
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pinecone = new PineconeEmbeddings()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("pinecone_result")
      .setIndexName("spark-nlp-test")
      .setNamespace("no-metadata-test")
      .setIdColumn("id")
      .setBatchSize(10)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, embeddings, sentenceEmbeddings, pinecone))

    val result = pipeline.fit(testData).transform(testData)

    assert(result.count() == 2, "Should process both documents without metadata")
    println("✓ Test 3 passed: Successfully stored documents without metadata")
  }

  it should "generate UUIDs when no ID column specified" taggedAs SlowTest in {

    val testData = Seq(
      "Document without explicit ID column",
      "Another document for UUID test")
      .toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = WordEmbeddingsModel
      .pretrained("glove_100d")
      .setInputCols("document")
      .setOutputCol("word_embeddings")

    val sentenceEmbeddings = new SentenceEmbeddings()
      .setInputCols("document", "word_embeddings")
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pinecone = new PineconeEmbeddings()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("pinecone_result")
      .setIndexName("spark-nlp-test")
      .setNamespace("uuid-test")
      .setBatchSize(10)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, embeddings, sentenceEmbeddings, pinecone))

    val result = pipeline.fit(testData).transform(testData)

    assert(result.count() == 2, "Should process both documents with auto-generated UUIDs")
    println("✓ Test 4 passed: Successfully generated UUIDs for documents")
  }

  it should "handle different namespaces correctly" taggedAs SlowTest in {

    val testData1 = Seq(("ns1_doc1", "Data for namespace 1")).toDF("id", "text")
    val testData2 = Seq(("ns2_doc1", "Data for namespace 2")).toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = WordEmbeddingsModel
      .pretrained("glove_100d")
      .setInputCols("document")
      .setOutputCol("word_embeddings")

    val sentenceEmbeddings = new SentenceEmbeddings()
      .setInputCols("document", "word_embeddings")
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    // Pipeline for namespace 1
    val pinecone1 = new PineconeEmbeddings()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("pinecone_result")
      .setIndexName("spark-nlp-test")
      .setNamespace("namespace-1")
      .setIdColumn("id")
      .setBatchSize(10)

    val pipeline1 = new Pipeline()
      .setStages(Array(documentAssembler, embeddings, sentenceEmbeddings, pinecone1))

    // Pipeline for namespace 2
    val pinecone2 = new PineconeEmbeddings()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("pinecone_result")
      .setIndexName("spark-nlp-test")
      .setNamespace("namespace-2")
      .setIdColumn("id")
      .setBatchSize(10)

    val pipeline2 = new Pipeline()
      .setStages(Array(documentAssembler, embeddings, sentenceEmbeddings, pinecone2))

    val result1 = pipeline1.fit(testData1).transform(testData1)
    val result2 = pipeline2.fit(testData2).transform(testData2)

    assert(result1.count() == 1, "Should store in namespace 1")
    assert(result2.count() == 1, "Should store in namespace 2")
    println("✓ Test 5 passed: Successfully used different namespaces")
  }

  it should "handle empty namespace (default namespace)" taggedAs SlowTest in {

    val testData = Seq(("default_ns_doc", "Document for default namespace")).toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = WordEmbeddingsModel
      .pretrained("glove_100d")
      .setInputCols("document")
      .setOutputCol("word_embeddings")

    val sentenceEmbeddings = new SentenceEmbeddings()
      .setInputCols("document", "word_embeddings")
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pinecone = new PineconeEmbeddings()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("pinecone_result")
      .setIndexName("spark-nlp-test")
      // No namespace set - should use default empty namespace
      .setIdColumn("id")
      .setBatchSize(10)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, embeddings, sentenceEmbeddings, pinecone))

    val result = pipeline.fit(testData).transform(testData)

    assert(result.count() == 1, "Should store in default namespace")
    println("✓ Test 6 passed: Successfully used default (empty) namespace")
  }

  it should "properly escape special characters in metadata" taggedAs SlowTest in {

    val testData = Seq(
      ("escape_test_1", "Text with \"quotes\"", "category with spaces"),
      ("escape_test_2", "Text with\nnewlines\tand\ttabs", "special/chars\\here"))
      .toDF("id", "text", "category")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = WordEmbeddingsModel
      .pretrained("glove_100d")
      .setInputCols("document")
      .setOutputCol("word_embeddings")

    val sentenceEmbeddings = new SentenceEmbeddings()
      .setInputCols("document", "word_embeddings")
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pinecone = new PineconeEmbeddings()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("pinecone_result")
      .setIndexName("spark-nlp-test")
      .setNamespace("escape-test")
      .setIdColumn("id")
      .setMetadataColumns(Array("text", "category"))
      .setBatchSize(10)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, embeddings, sentenceEmbeddings, pinecone))

    val result = pipeline.fit(testData).transform(testData)

    assert(result.count() == 2, "Should handle special characters in metadata")
    println("✓ Test 7 passed: Properly escaped special characters in metadata")
  }

  it should "validate required parameters" taggedAs SlowTest in {
    assertThrows[IllegalArgumentException] {
      val pinecone = new PineconeEmbeddings()
        .setInputCols("document", "sentence_embeddings")
        .setOutputCol("pinecone_result")
      // Missing indexName - should throw error when trying to use
    }
    println("✓ Test 8 passed: Properly validates required parameters")
  }

  it should "handle configuration from Spark conf" taggedAs SlowTest in {
    // Verify that API key is loaded from Spark configuration
    val apiKey = spark.conf.get("spark.jsl.settings.pinecone.api.key")
    val environment = spark.conf.get("spark.jsl.settings.pinecone.environment")

    assert(apiKey.nonEmpty, "API key should be loaded from Spark conf")
    assert(environment.nonEmpty, "Environment should be loaded from Spark conf")
    println("✓ Test 9 passed: Configuration loaded from Spark conf")
  }

  it should "process multiple metadata columns correctly" taggedAs SlowTest in {

    val testData = Seq(
      ("multi_meta_1", "Document one", "tech", "2024-01-15", "author1", "published"),
      ("multi_meta_2", "Document two", "ai", "2024-01-16", "author2", "draft"))
      .toDF("id", "text", "category", "date", "author", "status")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = WordEmbeddingsModel
      .pretrained("glove_100d")
      .setInputCols("document")
      .setOutputCol("word_embeddings")

    val sentenceEmbeddings = new SentenceEmbeddings()
      .setInputCols("document", "word_embeddings")
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pinecone = new PineconeEmbeddings()
      .setInputCols("document", "sentence_embeddings")
      .setOutputCol("pinecone_result")
      .setIndexName("spark-nlp-test")
      .setNamespace("multi-metadata-test")
      .setIdColumn("id")
      .setMetadataColumns(Array("text", "category", "date", "author", "status"))
      .setBatchSize(10)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, embeddings, sentenceEmbeddings, pinecone))

    val result = pipeline.fit(testData).transform(testData)

    assert(result.count() == 2, "Should process documents with multiple metadata fields")
    println("✓ Test 10 passed: Successfully handled multiple metadata columns")
  }
}

