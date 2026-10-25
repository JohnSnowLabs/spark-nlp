/*
 * Copyright 2017-2025 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.similarity

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotationUtils.AnnotationRow
import com.johnsnowlabs.nlp.AnnotatorType.SENTENCE_EMBEDDINGS
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.SparkException
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{col, explode}
import org.apache.spark.sql.types.{MetadataBuilder, StructField, StructType}
import org.scalatest.flatspec.AnyFlatSpec

class PairwiseVectorSimilarityTestSpec extends AnyFlatSpec {

  private val spark = ResourceHelper.spark

  private def seAnnotation(text: String, vec: Array[Float]): Annotation =
    Annotation(SENTENCE_EMBEDDINGS, 0, text.length - 1, text, Map("sentence" -> "0"), vec)

  private def makeDF(
      colAAnnots: Seq[Annotation],
      colBAnnots: Seq[Annotation],
      colAName: String = "emb_a",
      colBName: String = "emb_b") = {
    val meta = new MetadataBuilder().putString("annotatorType", SENTENCE_EMBEDDINGS).build()
    val schema = StructType(
      Seq(
        StructField(colAName, com.johnsnowlabs.nlp.Annotation.arrayType, nullable = false, meta),
        StructField(colBName, com.johnsnowlabs.nlp.Annotation.arrayType, nullable = false, meta)))
    val row = Row(colAAnnots.map(_.toRow()).toSeq, colBAnnots.map(_.toRow()).toSeq)
    spark.createDataFrame(spark.sparkContext.parallelize(Seq(row)), schema)
  }

  "PairwiseVectorSimilarity" should "compute cosine = 1.0 for identical vectors" taggedAs FastTest in {
    val vec = Array(1.0f, 0.0f, 0.0f)
    val df = makeDF(Seq(seAnnotation("hello", vec)), Seq(seAnnotation("hello", vec)))

    val result = new PairwiseVectorSimilarity()
      .setInputCols("emb_a", "emb_b")
      .setOutputCol("similarity")
      .setSimilarityMethod("cosine")
      .transform(df)
      .select(explode(col("similarity")).as("s"))
      .select(col("s.result").cast("double").as("score"))
      .collect()

    assert(result.length == 1)
    assert(math.abs(result.head.getAs[Double]("score") - 1.0) < 1e-9)
  }

  it should "compute cosine = 0.0 for orthogonal vectors" taggedAs FastTest in {
    val df = makeDF(
      Seq(seAnnotation("query", Array(1.0f, 0.0f))),
      Seq(seAnnotation("doc", Array(0.0f, 1.0f))))

    val score = new PairwiseVectorSimilarity()
      .setInputCols("emb_a", "emb_b")
      .setOutputCol("similarity")
      .transform(df)
      .select(explode(col("similarity")).as("s"))
      .select(col("s.result").cast("double").as("score"))
      .collect()
      .head
      .getAs[Double]("score")

    assert(math.abs(score) < 1e-9)
  }

  it should "compute cosine = -1.0 for opposite vectors" taggedAs FastTest in {
    val df = makeDF(
      Seq(seAnnotation("pos", Array(1.0f, 0.0f))),
      Seq(seAnnotation("neg", Array(-1.0f, 0.0f))))

    val score = new PairwiseVectorSimilarity()
      .setInputCols("emb_a", "emb_b")
      .setOutputCol("similarity")
      .transform(df)
      .select(explode(col("similarity")).as("s"))
      .select(col("s.result").cast("double").as("score"))
      .collect()
      .head
      .getAs[Double]("score")

    assert(math.abs(score - (-1.0)) < 1e-9)
  }

  it should "produce N×M annotations for multi-embedding inputs" taggedAs FastTest in {
    val df = makeDF(
      Seq(seAnnotation("q1", Array(1.0f, 0.0f)), seAnnotation("q2", Array(0.0f, 1.0f))),
      Seq(
        seAnnotation("d1", Array(1.0f, 0.0f)),
        seAnnotation("d2", Array(0.0f, 1.0f)),
        seAnnotation("d3", Array(-1.0f, 0.0f))))

    val count = new PairwiseVectorSimilarity()
      .setInputCols("emb_a", "emb_b")
      .setOutputCol("similarity")
      .transform(df)
      .select(explode(col("similarity")).as("s"))
      .count()

    assert(count == 6L) // 2 × 3
  }

  it should "populate sentence_a_text and sentence_b_text in metadata" taggedAs FastTest in {
    val df = makeDF(
      Seq(seAnnotation("my query", Array(1.0f, 0.0f))),
      Seq(seAnnotation("my document", Array(0.0f, 1.0f))))

    val row = new PairwiseVectorSimilarity()
      .setInputCols("emb_a", "emb_b")
      .setOutputCol("similarity")
      .transform(df)
      .select(explode(col("similarity")).as("s"))
      .select(
        col("s.metadata")("sentence_a_text").as("a_text"),
        col("s.metadata")("sentence_b_text").as("b_text"))
      .collect()
      .head

    assert(row.getAs[String]("a_text") == "my query")
    assert(row.getAs[String]("b_text") == "my document")
  }

  it should "have result equal to metadata similarity" taggedAs FastTest in {
    val df =
      makeDF(Seq(seAnnotation("a", Array(0.6f, 0.8f))), Seq(seAnnotation("b", Array(0.8f, 0.6f))))

    val row = new PairwiseVectorSimilarity()
      .setInputCols("emb_a", "emb_b")
      .setOutputCol("similarity")
      .transform(df)
      .select(explode(col("similarity")).as("s"))
      .select(
        col("s.result").cast("double").as("from_result"),
        col("s.metadata")("similarity").cast("double").as("from_meta"))
      .collect()
      .head

    assert(math.abs(row.getAs[Double]("from_result") - row.getAs[Double]("from_meta")) < 1e-15)
  }

  it should "return euclidean = 0.0 for identical vectors" taggedAs FastTest in {
    val vec = Array(1.0f, 2.0f, 3.0f)
    val df = makeDF(Seq(seAnnotation("x", vec)), Seq(seAnnotation("x", vec)))

    val score = new PairwiseVectorSimilarity()
      .setInputCols("emb_a", "emb_b")
      .setOutputCol("similarity")
      .setSimilarityMethod("euclidean")
      .transform(df)
      .select(explode(col("similarity")).as("s"))
      .select(col("s.result").cast("double").as("score"))
      .collect()
      .head
      .getAs[Double]("score")

    assert(math.abs(score) < 1e-9)
  }

  it should "return negative euclidean for distinct vectors" taggedAs FastTest in {
    val df =
      makeDF(Seq(seAnnotation("a", Array(1.0f, 0.0f))), Seq(seAnnotation("b", Array(0.0f, 1.0f))))

    val score = new PairwiseVectorSimilarity()
      .setInputCols("emb_a", "emb_b")
      .setOutputCol("similarity")
      .setSimilarityMethod("euclidean")
      .transform(df)
      .select(explode(col("similarity")).as("s"))
      .select(col("s.result").cast("double").as("score"))
      .collect()
      .head
      .getAs[Double]("score")

    assert(score < 0.0)
    // expected: -sqrt(2) ≈ -1.4142
    assert(math.abs(score - (-math.sqrt(2.0))) < 1e-9)
  }

  it should "return dotProduct scores correctly" taggedAs FastTest in {
    val df =
      makeDF(Seq(seAnnotation("a", Array(2.0f, 3.0f))), Seq(seAnnotation("b", Array(4.0f, 5.0f))))

    val score = new PairwiseVectorSimilarity()
      .setInputCols("emb_a", "emb_b")
      .setOutputCol("similarity")
      .setSimilarityMethod("dotProduct")
      .transform(df)
      .select(explode(col("similarity")).as("s"))
      .select(col("s.result").cast("double").as("score"))
      .collect()
      .head
      .getAs[Double]("score")

    // 2*4 + 3*5 = 23
    assert(math.abs(score - 23.0) < 1e-9)
  }

  it should "return 0.0 for cosine with a zero vector" taggedAs FastTest in {
    val df = makeDF(
      Seq(seAnnotation("zero", Array(0.0f, 0.0f))),
      Seq(seAnnotation("nonzero", Array(1.0f, 0.0f))))

    val score = new PairwiseVectorSimilarity()
      .setInputCols("emb_a", "emb_b")
      .setOutputCol("similarity")
      .transform(df)
      .select(explode(col("similarity")).as("s"))
      .select(col("s.result").cast("double").as("score"))
      .collect()
      .head
      .getAs[Double]("score")

    assert(score == 0.0)
  }

  it should "throw a clear error for mismatched embedding dimensions" taggedAs FastTest in {
    val df = makeDF(
      Seq(seAnnotation("short", Array(1.0f, 0.0f))),
      Seq(seAnnotation("long", Array(1.0f, 0.0f, 0.0f))))

    val exception = intercept[SparkException] {
      new PairwiseVectorSimilarity()
        .setInputCols("emb_a", "emb_b")
        .setOutputCol("similarity")
        .transform(df)
        .select(explode(col("similarity")).as("s"))
        .collect()
    }

    assert(exception.toString.contains("requires equal embedding dimensions"))
  }

  it should "throw when setInputCols receives only one column" taggedAs FastTest in {
    assertThrows[IllegalArgumentException] {
      new PairwiseVectorSimilarity().setInputCols("only_one")
    }
  }

  it should "throw UnsupportedOperationException from annotate()" taggedAs FastTest in {
    assertThrows[UnsupportedOperationException] {
      new PairwiseVectorSimilarity().annotate(Seq.empty)
    }
  }
}
