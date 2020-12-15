package com.johnsnowlabs.nlp

import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{col, concat, lit}
import org.scalatest.Matchers._
import org.scalatest._

import scala.collection.Map
import scala.language.reflectiveCalls

class DocumentAssemblerTestSpec extends FlatSpec {
  def fixture = new {
    val text = ContentProvider.englishPhrase
    val df = AnnotatorBuilder.withDocumentAssembler(DataBuilder.basicDataBuild(text))
    val assembledDoc = df
      .select("document")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }
  }

  def nullFixture = new {
    val corrupted = DataBuilder.loadParquetDataset("src/test/resources/doc-assembler/corrupted_data.parquet")

    val preprocessed =
      corrupted
        .select(col("id"), concat(col("title"), lit(" "), col("content")))
        .withColumnRenamed("concat(title,  , content)", "text")

    val processed = AnnotatorBuilder.withDocumentAssembler(preprocessed, "shrink")

    val assembledDoc = processed
      .select("document")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }
  }

  "A DocumentAssembler" should "annotate with the correct indexes" in {
    val f = fixture
    f.text.head should equal (f.text(f.assembledDoc.head.begin))
    f.text.last should equal (f.text(f.assembledDoc.head.end))
  }

  "A DocumentAssembler" should "produce an empty annotation in a pipeline of null texts" in {
    val f = nullFixture

    Annotation(AnnotatorType.DOCUMENT, 0, "".length - 1, "", Map.empty[String, String], Array.emptyFloatArray)
      .begin should equal (f.assembledDoc.head.begin)

    Annotation(AnnotatorType.DOCUMENT, 0, "".length - 1, "", Map.empty[String, String], Array.emptyFloatArray)
      .end should equal (f.assembledDoc.head.end)
  }
}