package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, SparkAccessor}
import org.apache.spark.sql.Row
import org.scalatest.Matchers.{convertToAnyShouldWrapper, equal}
import org.scalatest._

import scala.language.reflectiveCalls

trait DocumentNormalizerBehaviors extends FlatSpec {

  val DOC_NORMALIZER_BASE_DIR = "src/test/resources/doc-normalizer"

  def fixtureFilesHTML(action: String, patterns: Array[String]) = {

    import SparkAccessor.spark.implicits._

    val dataset =
      SparkAccessor.spark.sparkContext
        .wholeTextFiles(s"$DOC_NORMALIZER_BASE_DIR/html-docs")
        .toDF("filename", "text")
        .select("text")

    val annotated =
      AnnotatorBuilder
        .withDocumentNormalizer(
          dataset = dataset,
          action = action,
          actionPatterns = patterns)

    val normalizedDoc: Array[Annotation] = annotated
      .select("normalizedDocument")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }

    normalizedDoc
  }

  def fixtureFilesXML(action: String, patterns: Array[String]) = {

    import SparkAccessor.spark.implicits._

    val dataset =
      SparkAccessor.spark.sparkContext
        .wholeTextFiles(s"$DOC_NORMALIZER_BASE_DIR/xml-docs")
        .toDF("filename", "text")
        .select("text")

    val annotated = AnnotatorBuilder.withDocumentNormalizer(dataset = dataset, actionPatterns = Array("<[^>]*>"))

    val normalizedDoc: Array[Annotation] = annotated
      .select("normalizedDocument")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }

    normalizedDoc
  }

  "A DocumentNormalizer" should "annotate with the correct indexes removing all tags" in {

    val action = "clean_up"
    val patterns = Array("<[^>]*>") // all tags

    val f = fixtureFilesHTML(action, patterns)

    0 should equal (f.head.begin)

    674 should equal (f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes removing all specified p tags content" in {

    val action = "clean_up"
    val tag = "p"
    val patterns = Array("<"+tag+"(.+?)>(.+?)<\\/"+tag+">")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal (f.head.begin)
    605 should equal (f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes removing all specified h1 tags content" in {

    val action = "clean_up"
    val tag = "h1"
    val patterns = Array("<"+tag+"(.*?)>(.*?)<\\/"+tag+">")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal (f.head.begin)
    1140 should equal (f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes removing emails" in {

    val action = "clean_up"
    val patterns = Array("([^.@\\s]+)(\\.[^.@\\s]+)*@([^.@\\s]+\\.)+([^.@\\s]+)")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal (f.head.begin)
    1212 should equal (f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes removing ages" in {

    val action = "clean_up"
    val patterns = Array("\\d+(?=[\\s]?year)", "(aged)[\\s]?\\d+")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal (f.last.begin)
    409 should equal (f.last.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes extracting all a tags content" in {

    val action = "extract"
    val tag = "a"
    val patterns = Array(s"<(?!\\/?$tag(?=>|\\s.*>))\\/?.*?>")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal (f.head.begin)
    869 should equal (f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes extracting all div tags content" in {

    val action = "extract"
    val tag = "div"
    val patterns = Array(s"<(?!\\/?$tag(?=>|\\s.*>))\\/?.*?>")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal (f.head.begin)
    925 should equal (f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes extracting all b tags content" in {

    val action = "extract"
    val tag = "b"
    val patterns = Array(s"<(?!\\/?$tag(?=>|\\s.*>))\\/?.*?>")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal (f.head.begin)
    674 should equal (f.head.end)
  }
}
