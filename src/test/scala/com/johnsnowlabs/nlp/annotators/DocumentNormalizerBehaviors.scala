package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, DataBuilder, SparkAccessor}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.Matchers.{convertToAnyShouldWrapper, equal}
import org.scalatest._

import scala.io.Source
import scala.language.reflectiveCalls

trait DocumentNormalizerBehaviors extends FlatSpec {

  val DOC_NORMALIZER_BASE_DIR = "src/test/resources/doc-normalizer"

  private def loadDocNormalizerDataset(path: String): Dataset[Row] = {
    DataBuilder
      .basicDataBuild(
        Source.fromFile(path).getLines().mkString)
  }

  def fixture = new {
    val scrapedTextDS: Dataset[Row] =
      loadDocNormalizerDataset(s"$DOC_NORMALIZER_BASE_DIR/scraped_text_small.txt")

    val annotated = AnnotatorBuilder.withDocumentNormalizerPipeline(scrapedTextDS)

    val normalizedDoc: Array[Annotation] = annotated
      .select("normalizedDocument")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }
  }

  def fixtureMultipleDocs = new {
    import SparkAccessor.spark.implicits._
    val dataset =
      SparkAccessor.spark.sparkContext
        .wholeTextFiles(s"$DOC_NORMALIZER_BASE_DIR/webpage-samples")
        .toDF("filename", "text")
        .select("text")

    val annotated = AnnotatorBuilder.withDocumentNormalizerPipelineForHTML(dataset)

    val normalizedDoc: Array[Annotation] = annotated
      .select("normalizedDocument")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }
  }

  "A DocumentNormalizer" should "annotate with the correct indexes" in {
    val f = fixture

    0 should equal (f.normalizedDoc.head.begin)

    57 should equal (f.normalizedDoc.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct metadata" in {
    val f = fixture

    Map("sentence" -> "0") should equal (f.normalizedDoc.head.metadata)
  }

  "A DocumentNormalizer" should "annotate multiple tagged documents with the correct indexes" in {
    val f = fixtureMultipleDocs

    0 should equal (f.normalizedDoc.head.begin)

    13410 should equal (f.normalizedDoc.head.end)
  }

  "A DocumentNormalizer" should "annotate multiple tagged documents with the correct metadata" in {
    val f = fixtureMultipleDocs

    Map("sentence" -> "0") should equal (f.normalizedDoc.head.metadata)
  }

  "A DocumentNormalizer" should "annotate the correct number of sample documents" in {
    val f = fixtureMultipleDocs

    3 should equal (f.annotated.count)
  }
}
