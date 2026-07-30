package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import org.apache.spark.sql.Row
import org.scalatest.flatspec.AnyFlatSpec

class AnnotationRowUtilsTest extends AnyFlatSpec with SparkSessionTest {

  "AnnotationRowUtils.extractAnnotationRows" should "return rows from scala sequences and arrays and tolerate nulls" in {
    val annotationA =
      Row("document", 0, 3, "alpha", Map.empty[String, String], Array.empty[Float])
    val annotationB = Row("document", 4, 7, "beta", Map("k" -> "v"), Array(1.0f))

    val rowWithSeq = Row(Seq(annotationA, annotationB))
    val rowWithArray = Row(Array(annotationA, annotationB))
    val rowWithNull = Row(null)

    assert(
      AnnotationRowUtils.extractAnnotationRows(rowWithSeq, 0) == Vector(annotationA, annotationB))
    assert(
      AnnotationRowUtils
        .extractAnnotationRows(rowWithArray, 0) == Vector(annotationA, annotationB))
    assert(AnnotationRowUtils.extractAnnotationRows(rowWithNull, 0).isEmpty)
  }

  it should "throw for non annotation-array values" in {
    val invalidRow = Row("not-an-array")

    val error = intercept[IllegalArgumentException] {
      AnnotationRowUtils.extractAnnotationRows(invalidRow, 0)
    }

    assert(error.getMessage.contains("Expected annotation array at column 0"))
  }

  it should "convert annotations to rows" in {
    val annotation = com.johnsnowlabs.nlp.Annotation(
      annotatorType = "document",
      begin = 0,
      end = 4,
      result = "alpha",
      metadata = Map("sentence" -> "0"),
      embeddings = Array(1.0f, 2.0f))

    val row = AnnotationRowUtils.annotationToRow(annotation)

    assert(row.getString(0) == "document")
    assert(row.getInt(1) == 0)
    assert(row.getInt(2) == 4)
    assert(row.getString(3) == "alpha")
    assert(row.getMap[String, String](4) == Map("sentence" -> "0"))
    assert(row.getAs[Array[Float]](5).sameElements(Array(1.0f, 2.0f)))
  }

  it should "round-trip annotations with primitive-array embeddings" in {
    val original = com.johnsnowlabs.nlp.Annotation(
      annotatorType = "word_embeddings",
      begin = 0,
      end = 4,
      result = "alpha",
      metadata = Map("sentence" -> "0", "token" -> "alpha"),
      embeddings = Array(1.0f, 2.0f))

    val row = AnnotationRowUtils.annotationToRow(original)
    val restored = com.johnsnowlabs.nlp.Annotation(row)

    assert(restored == original)
    assert(!(restored.embeddings eq original.embeddings))
  }

  it should "convert annotations with sequence embeddings" in {
    val row = Row(
      "word_embeddings",
      0,
      4,
      "alpha",
      Map("sentence" -> "0", "token" -> "alpha"),
      Seq(1.0f, 2.0f))

    val annotation = com.johnsnowlabs.nlp.Annotation(row)

    assert(annotation.embeddings.sameElements(Array(1.0f, 2.0f)))
  }
}
