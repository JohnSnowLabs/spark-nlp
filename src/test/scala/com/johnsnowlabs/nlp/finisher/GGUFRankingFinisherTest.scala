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

package com.johnsnowlabs.nlp.finisher

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, ContentProvider}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

class GGUFRankingFinisherTest extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._
  // Mock data to simulate AutoGGUFReranker output
  def createMockRerankerOutput(): DataFrame = {
    val spark = ResourceHelper.spark

    val documents = Seq(
      ("A man is eating food.", 0.85, "A man is eating pasta."),
      ("A man is eating a piece of bread.", 0.72, "A man is eating pasta."),
      ("The girl is carrying a baby.", 0.15, "A man is eating pasta."),
      ("A man is riding a horse.", 0.28, "A man is eating pasta."),
      ("A young girl is playing violin.", 0.05, "A man is eating pasta."))

    val mockAnnotations = documents.map { case (text, score, query) =>
      Row(
        AnnotatorType.DOCUMENT,
        0,
        text.length - 1,
        text,
        Map("relevance_score" -> score.toString, "query" -> query),
        Array.empty[Float])
    }

    val rows = Seq(Row(mockAnnotations))
    val annotationSchema = StructType(
      Array(
        StructField("annotatorType", StringType, nullable = false),
        StructField("begin", IntegerType, nullable = false),
        StructField("end", IntegerType, nullable = false),
        StructField("result", StringType, nullable = false),
        StructField("metadata", MapType(StringType, StringType), nullable = false),
        StructField("embeddings", ArrayType(FloatType), nullable = false)))
    val schema = StructType(
      Array(StructField("reranked_documents", ArrayType(annotationSchema), nullable = false)))

    spark.createDataFrame(spark.sparkContext.parallelize(rows), schema)
  }

  "GGUFRankingFinisher with default settings" should "process documents and add rank metadata" taggedAs FastTest in {
    val mockData = createMockRerankerOutput()

    val finisher = new GGUFRankingFinisher()
      .setInputCols("reranked_documents")
      .setOutputCol("ranked_documents")

    val result = finisher.transform(mockData)

    assert(result.columns.contains("ranked_documents"))

    // Get the ranked documents
    val rankedDocs =
      result.select("ranked_documents").rdd.map(_.getAs[Seq[Row]](0)).collect().head

    assert(rankedDocs.length == 5)

    // Check that results are sorted by relevance score in descending order
    val scores =
      rankedDocs.map(_.getAs[Map[String, String]]("metadata")("relevance_score").toDouble)
    assert(scores.zip(scores.tail).forall { case (a, b) => a >= b })

    // Check that rank metadata is added
    val ranks = rankedDocs.map(_.getAs[Map[String, String]]("metadata")("rank").toInt)
    assert(ranks == Seq(1, 2, 3, 4, 5))
  }

  "GGUFRankingFinisher with topK" should "return only top k results" taggedAs FastTest in {
    val mockData = createMockRerankerOutput()

    val finisher = new GGUFRankingFinisher()
      .setInputCols("reranked_documents")
      .setOutputCol("ranked_documents")
      .setTopK(3)

    val result = finisher.transform(mockData)

    // Should have only 1 row since all annotations are in a single row
    assert(result.count() == 1)

    val rankedDocs =
      result.select("ranked_documents").rdd.map(_.getAs[Seq[Row]](0)).collect().head
    assert(rankedDocs.length == 3)

    // Check that we get the top 3 scores
    val scores =
      rankedDocs.map(_.getAs[Map[String, String]]("metadata")("relevance_score").toDouble)
    assert(scores.length == 3)
    assert(scores.contains(0.85))
    assert(scores.contains(0.72))
    assert(scores.contains(0.28))

    // Check ranks are correct
    val ranks = rankedDocs.map(_.getAs[Map[String, String]]("metadata")("rank").toInt)
    assert(ranks == Seq(1, 2, 3))
  }

  "GGUFRankingFinisher with threshold" should "filter by minimum relevance score" taggedAs FastTest in {
    val mockData = createMockRerankerOutput()

    val finisher = new GGUFRankingFinisher()
      .setInputCols("reranked_documents")
      .setOutputCol("ranked_documents")
      .setMinRelevanceScore(0.3)

    val result = finisher.transform(mockData)

    val rankedDocs =
      result.select("ranked_documents").rdd.map(_.getAs[Seq[Row]](0)).collect().head
    assert(rankedDocs.length == 2) // Only scores >= 0.3 (0.85 and 0.72)

    val scores =
      rankedDocs.map(_.getAs[Map[String, String]]("metadata")("relevance_score").toDouble)
    assert(scores.forall(_ >= 0.3))
  }

  "GGUFRankingFinisher with min-max scaling" should "normalize scores to 0-1 range" taggedAs FastTest in {
    val mockData = createMockRerankerOutput()

    val finisher = new GGUFRankingFinisher()
      .setInputCols("reranked_documents")
      .setOutputCol("ranked_documents")
      .setMinMaxScaling(true)

    val result = finisher.transform(mockData)

    val rankedDocs =
      result.select("ranked_documents").rdd.map(_.getAs[Seq[Row]](0)).collect().head
    val scores =
      rankedDocs.map(_.getAs[Map[String, String]]("metadata")("relevance_score").toDouble)

    // Check that scores are between 0 and 1
    assert(scores.forall(score => score >= 0.0 && score <= 1.0))

    // Check that we have both min (0.0) and max (1.0) values
    assert(scores.contains(1.0)) // Max original score should be 1.0
    assert(scores.contains(0.0)) // Min original score should be 0.0
  }

  "GGUFRankingFinisher with combined filters" should "apply topK, threshold, and scaling together" taggedAs FastTest in {
    val mockData = createMockRerankerOutput()

    val finisher = new GGUFRankingFinisher()
      .setInputCols("reranked_documents")
      .setOutputCol("ranked_documents")
      .setTopK(2)
      .setMinRelevanceScore(0.1) // After scaling, this should filter some results
      .setMinMaxScaling(true)

    val result = finisher.transform(mockData)

    val rankedDocs =
      result.select("ranked_documents").rdd.map(_.getAs[Seq[Row]](0)).collect().head

    // Should have at most 2 results due to topK
    assert(rankedDocs.length <= 2)

    val scores =
      rankedDocs.map(_.getAs[Map[String, String]]("metadata")("relevance_score").toDouble)

    // All scores should be >= 0.1 and <= 1.0
    assert(scores.forall(score => score >= 0.1 && score <= 1.0))

    // Results should be sorted descending
    assert(scores.zip(scores.tail).forall { case (a, b) => a >= b })

    // Check that ranks are correct
    val ranks = rankedDocs.map(_.getAs[Map[String, String]]("metadata")("rank").toInt)
    assert(ranks == (1 to rankedDocs.length).toSeq)
  }

  "GGUFRankingFinisher" should "handle empty input" taggedAs FastTest in {
    val spark = ResourceHelper.spark

    val emptyRows = Seq(Row(Array.empty[Row]))
    val annotationSchema = StructType(
      Array(
        StructField("annotatorType", StringType, nullable = false),
        StructField("begin", IntegerType, nullable = false),
        StructField("end", IntegerType, nullable = false),
        StructField("result", StringType, nullable = false),
        StructField("metadata", MapType(StringType, StringType), nullable = false),
        StructField("embeddings", ArrayType(FloatType), nullable = false)))
    val schema = StructType(
      Array(StructField("reranked_documents", ArrayType(annotationSchema), nullable = false)))

    val emptyData = spark.createDataFrame(spark.sparkContext.parallelize(emptyRows), schema)

    val finisher = new GGUFRankingFinisher()
      .setInputCols("reranked_documents")
      .setOutputCol("ranked_documents")

    val result = finisher.transform(emptyData)

    // Since we now filter out empty rows, the result should have no rows
    assert(result.count() == 0)
  }

  "GGUFRankingFinisher" should "preserve query information in metadata" taggedAs FastTest in {
    val mockData = createMockRerankerOutput()

    val finisher = new GGUFRankingFinisher()
      .setInputCols("reranked_documents")
      .setOutputCol("ranked_documents")

    val result = finisher.transform(mockData)

    val rankedDocs =
      result.select("ranked_documents").rdd.map(_.getAs[Seq[Row]](0)).collect().head

    // Check that query information is preserved in metadata
    rankedDocs.foreach { doc =>
      val metadata = doc.getAs[Map[String, String]]("metadata")
      assert(metadata.contains("query"))
      assert(metadata("query") == "A man is eating pasta.")
    }
  }

  "GGUFRankingFinisher" should "handle documents with missing relevance scores" taggedAs FastTest in {
    val spark = ResourceHelper.spark

    val documents = Seq(
      ("A man is eating food.", Some("0.85"), "A man is eating pasta."),
      ("A man is eating a piece of bread.", None, "A man is eating pasta."), // Missing score
      ("The girl is carrying a baby.", Some("0.15"), "A man is eating pasta."))

    val testAnnotations: Seq[Row] = documents.map { case (text, scoreOpt, query) =>
      val metadata = Map("query" -> query) ++ scoreOpt.map("relevance_score" -> _).toMap
      Row(AnnotatorType.DOCUMENT, 0, text.length - 1, text, metadata, Array.empty[Float])
    }

    val rows = Seq(Row(testAnnotations))
    val annotationSchema = StructType(
      Array(
        StructField("annotatorType", StringType, nullable = false),
        StructField("begin", IntegerType, nullable = false),
        StructField("end", IntegerType, nullable = false),
        StructField("result", StringType, nullable = false),
        StructField("metadata", MapType(StringType, StringType), nullable = false),
        StructField("embeddings", ArrayType(FloatType), nullable = false)))
    val schema = StructType(
      Array(StructField("reranked_documents", ArrayType(annotationSchema), nullable = false)))

    val testData = spark.createDataFrame(spark.sparkContext.parallelize(rows), schema)

    val finisher = new GGUFRankingFinisher()
      .setInputCols("reranked_documents")
      .setOutputCol("ranked_documents")

    val result = finisher.transform(testData)

    val rankedDocs =
      result.select("ranked_documents").rdd.map(_.getAs[Seq[Row]](0)).collect().head
    assert(rankedDocs.length == 3)

    // Document with missing score should get 0.0 and be ranked last
    val scores =
      rankedDocs.map(_.getAs[Map[String, String]]("metadata")("relevance_score").toDouble)
    assert(scores.last == 0.0) // Missing score becomes 0.0
  }

  "GGUFRankingFinisher with topK across multiple rows" should "filter out empty rows and return only top k globally" taggedAs FastTest in {
    val spark = ResourceHelper.spark

    val documents = Seq(
      ("A man is eating food.", 0.85, "A man is eating pasta."),
      ("A man is eating a piece of bread.", 0.72, "A man is eating pasta."),
      ("The girl is carrying a baby.", 0.15, "A man is eating pasta."),
      ("A man is riding a horse.", 0.28, "A man is eating pasta."),
      ("A young girl is playing violin.", 0.05, "A man is eating pasta."))

    // Create individual rows, each with one annotation (simulating real usage)
    val testRows: Seq[Row] = documents.map { case (text, score, query) =>
      val annotation = Row(
        AnnotatorType.DOCUMENT,
        0,
        text.length - 1,
        text,
        Map("relevance_score" -> score.toString, "query" -> query),
        Array.empty[Float])
      Row(Array(annotation))
    }

    val annotationSchema = StructType(
      Array(
        StructField("annotatorType", StringType, nullable = false),
        StructField("begin", IntegerType, nullable = false),
        StructField("end", IntegerType, nullable = false),
        StructField("result", StringType, nullable = false),
        StructField("metadata", MapType(StringType, StringType), nullable = false),
        StructField("embeddings", ArrayType(FloatType), nullable = false)))
    val schema = StructType(
      Array(StructField("reranked_documents", ArrayType(annotationSchema), nullable = false)))

    val testData = spark.createDataFrame(spark.sparkContext.parallelize(testRows), schema)

    val finisher = new GGUFRankingFinisher()
      .setInputCols("reranked_documents")
      .setOutputCol("ranked_documents")
      .setTopK(3)

    val result = finisher.transform(testData)

    // Should have only 3 rows (top-3 globally)
    assert(result.count() == 3)

    val allAnnotations = result.select("ranked_documents").collect().flatMap(_.getSeq[Row](0))
    val scores =
      allAnnotations.map(_.getAs[Map[String, String]]("metadata")("relevance_score").toDouble)
    val ranks = allAnnotations.map(_.getAs[Map[String, String]]("metadata")("rank").toInt)

    // Should have exactly 3 documents with scores 0.85, 0.72, 0.28
    assert(scores.length == 3)
    assert(scores.contains(0.85))
    assert(scores.contains(0.72))
    assert(scores.contains(0.28))

    // Ranks should be 1, 2, 3
    assert(ranks.sorted sameElements Array(1, 2, 3))
  }
}
