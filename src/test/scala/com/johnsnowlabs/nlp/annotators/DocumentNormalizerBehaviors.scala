package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, DataBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.Matchers.{convertToAnyShouldWrapper, equal}
import org.scalatest._

import scala.io.Source
import scala.language.reflectiveCalls

trait DocumentNormalizerBehaviors extends FlatSpec {

  val path = ""

  private def loadDocNormalizerDataset(path: String): Dataset[Row] = {
    DataBuilder
      .basicDataBuild(
        Source.fromFile("src/test/resources/doc-normalizer/scraped_text_small.txt").getLines().mkString)
  }

  def fixture = new {
    val scrapedTextDS: Dataset[Row] = loadDocNormalizerDataset("doc-normalizer/scraped_text_small.txt")

    val df = AnnotatorBuilder.withDocumentNormalizerPipeline(scrapedTextDS)

    val normalizedDoc: Array[Annotation] = df
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
}
