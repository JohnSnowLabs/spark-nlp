/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.asc
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.must.Matchers._
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

import scala.language.reflectiveCalls

trait DocumentNormalizerBehaviors extends AnyFlatSpec {

  val DOC_NORMALIZER_BASE_DIR = "src/test/resources/doc-normalizer"

  def runLookaroundDocNormPipeline(
      action: String,
      patterns: Array[String],
      replacement: String = " ") = {
    val dataset =
      SparkAccessor.spark
        .createDataFrame(
          List(
            (1, "10.2"),
            (2, "9,53"),
            (3, "11.01 mg"),
            (4, "mg 11.01"),
            (5, "14,220"),
            (6, "Amoxiciline 4,5 mg for $10.35; Ibuprofen 5,5mg for $9.00.")))
        .toDF("id", "text")

    val annotated =
      AnnotatorBuilder
        .withDocumentNormalizer(
          dataset = dataset,
          action = action,
          patterns = patterns,
          replacement = replacement)

    val normalizedDoc: Array[Annotation] = annotated
      .select("normalizedDocument")
      .collect
      .flatMap {
        _.getSeq[Row](0)
      }
      .map {
        Annotation(_)
      }

    normalizedDoc
  }

  def fixtureFilesHTML(action: String, patterns: Array[String], replacement: String = " ") = {

    import SparkAccessor.spark.implicits._

    val dataset =
      SparkAccessor.spark.sparkContext
        .wholeTextFiles(s"$DOC_NORMALIZER_BASE_DIR/html-docs")
        .toDF("filename", "text")
        .orderBy(asc("filename"))
        .select("text")

    val annotated =
      AnnotatorBuilder
        .withDocumentNormalizer(
          dataset = dataset,
          action = action,
          patterns = patterns,
          replacement = replacement)

    val normalizedDoc: Array[Annotation] = annotated
      .select("normalizedDocument")
      .collect
      .flatMap {
        _.getSeq[Row](0)
      }
      .map {
        Annotation(_)
      }

    normalizedDoc
  }

  def fixtureFilesXML(action: String, patterns: Array[String]) = {

    import SparkAccessor.spark.implicits._

    val dataset =
      SparkAccessor.spark.sparkContext
        .wholeTextFiles(s"$DOC_NORMALIZER_BASE_DIR/xml-docs")
        .toDF("filename", "text")
        .orderBy(asc("filename"))
        .select("text")

    val annotated =
      AnnotatorBuilder
        .withDocumentNormalizer(dataset = dataset, action = action, patterns = patterns)

    val normalizedDoc: Array[Annotation] = annotated
      .select("normalizedDocument")
      .collect
      .flatMap {
        _.getSeq[Row](0)
      }
      .map {
        Annotation(_)
      }

    normalizedDoc
  }

  def fixtureFilesJSON(action: String, patterns: Array[String]) = {

    import SparkAccessor.spark.implicits._

    val dataset =
      SparkAccessor.spark.sparkContext
        .wholeTextFiles(s"$DOC_NORMALIZER_BASE_DIR/json-docs")
        .toDF("filename", "text")
        .orderBy(asc("filename"))
        .select("text")

    val annotated =
      AnnotatorBuilder
        .withDocumentNormalizer(dataset = dataset, action = action, patterns = patterns)

    val normalizedDoc: Array[Annotation] = annotated
      .select("normalizedDocument")
      .collect
      .flatMap {
        _.getSeq[Row](0)
      }
      .map {
        Annotation(_)
      }

    normalizedDoc
  }

  "A DocumentNormalizer" should "annotate replacing , to . using iterative positive lookahead" taggedAs FastTest in {
    val action = "lookaround"
    val patterns = Array(".*\\d+(\\,)\\d+(?=\\s?mg).*")
    val replacement = "."

    val f = runLookaroundDocNormPipeline(action, patterns, replacement)(
      5
    ) // Amoxiciline 4,5 mg for $10.35; Ibuprofen 5,5mg for $9.00.

    0 should equal(f.begin)
    55 should equal(f.end)
  }

  "A DocumentNormalizer" should "annotate replacing . to , using positive lookahead" taggedAs FastTest in {
    val action = "lookaround"
    val patterns = Array(".*\\d+(\\.)\\d+(?= mg).*")
    val replacement = ","

    val f = runLookaroundDocNormPipeline(action, patterns, replacement)(2) // 11,01 mg

    println(f)

    0 should equal(f.begin)
    7 should equal(f.end)
  }

  "A DocumentNormalizer" should "annotate replacing . to , using positive lookbehind" taggedAs FastTest in {
    val action = "lookaround"
    val patterns = Array(".*(?<=mg )\\d+(\\.)\\d+.*")
    val replacement = ","

    val f = runLookaroundDocNormPipeline(action, patterns, replacement)(3) // mg 11,01

    0 should equal(f.begin)
    7 should equal(f.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes cleaning up all HTML tags" taggedAs FastTest in {

    val action = "clean"
    val patterns = Array("<[^>]*>")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal(f.head.begin)

    675 should equal(f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes cleaning up all specified p HTML tags content" taggedAs FastTest in {

    val action = "clean"
    val tag = "p"
    val patterns = Array("<" + tag + "(.+?)>(.+?)<\\/" + tag + ">")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal(f.head.begin)
    605 should equal(f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes cleaning up all specified h1 HTML tags content" taggedAs FastTest in {

    val action = "clean"
    val tag = "h1"
    val patterns = Array("<" + tag + "(.*?)>(.*?)<\\/" + tag + ">")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal(f.head.begin)
    1140 should equal(f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes cleaning up all specified br HTML tags content" taggedAs FastTest in {

    val action = "clean"
    val tag = "br"
    val patterns = Array("<" + tag + "(.*?)>")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal(f.last.begin)
    409 should equal(f.last.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes cleaning up emails" taggedAs FastTest in {

    val action = "clean"
    val patterns = Array("([^.@\\s]+)(\\.[^.@\\s]+)*@([^.@\\s]+\\.)+([^.@\\s]+)")
    val replacement = "***OBFUSCATED PII***"

    val f = fixtureFilesHTML(action, patterns, replacement)

    0 should equal(f.last.begin)
    410 should equal(f.last.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes cleaning up ages" taggedAs FastTest in {

    val action = "clean"
    val patterns = Array("\\d+(?=[\\s]?year)", "(aged)[\\s]?\\d+")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal(f.last.begin)
    399 should equal(f.last.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes cleaning up all a HTML tags content" taggedAs FastTest in {

    val action = "clean"
    val tag = "a"
    val patterns = Array(s"<(?!\\/?$tag(?=>|\\s.*>))\\/?.*?>")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal(f.head.begin)
    871 should equal(f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes cleaning up all div HTML tags" taggedAs FastTest in {

    val action = "clean"
    val tag = "div"
    val patterns = Array(s"<(?!\\/?$tag(?=>|\\s.*>))\\/?.*?>")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal(f.head.begin)
    926 should equal(f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes cleaning up all b HTML tags content" taggedAs FastTest in {

    val action = "clean"
    val tag = "b"
    val patterns = Array(s"<(?!\\/?$tag(?=>|\\s.*>))\\/?.*?>")

    val f = fixtureFilesHTML(action, patterns)

    0 should equal(f.head.begin)
    675 should equal(f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes extracting all div HTML tags contents" taggedAs FastTest in {

    val action = "extract"
    val tag = "div"
    val patterns = Array(tag)

    val f = fixtureFilesHTML(action, patterns)

    0 should equal(f.head.begin)
    1335 should equal(f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes extracting all p HTML tags contents" taggedAs FastTest in {

    val action = "extract"
    val tag = "p"
    val patterns = Array(tag)

    val f = fixtureFilesHTML(action, patterns)

    0 should equal(f.head.begin)
    574 should equal(f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes extracting all h1 HTML tags contents" taggedAs FastTest in {

    val action = "extract"
    val tag = "h1"
    val patterns = Array(tag)

    val f = fixtureFilesHTML(action, patterns)

    0 should equal(f.head.begin)
    37 should equal(f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes extracting XML streetAddressLine tag contents" taggedAs FastTest in {

    val action = "extract"
    val tag = "streetAddressLine"
    val patterns = Array(tag)

    val f = fixtureFilesXML(action, patterns)

    0 should equal(f.head.begin)
    301 should equal(f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes extracting XML name tag contents" taggedAs FastTest in {

    val action = "extract"
    val tag = "name"
    val patterns = Array(tag)

    val f = fixtureFilesXML(action, patterns)

    0 should equal(f.head.begin)
    638 should equal(f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes extracting XML family tag contents" taggedAs FastTest in {

    val action = "extract"
    val tag = "family"
    val patterns = Array(tag)

    val f = fixtureFilesXML(action, patterns)

    0 should equal(f.head.begin)
    59 should equal(f.head.end)
  }

  "A DocumentNormalizer" should "annotate with the correct indexes cleaning up JSON author field contents" taggedAs FastTest in {

    val action = "clean"
    val tag = "author"
    val patterns = Array(s""""$tag": "([^"]+)",""")

    val f = fixtureFilesJSON(action, patterns)

    0 should equal(f.head.begin)
    396 should equal(f.head.end)
  }

  "A DocumentNormalizer" should "annotate with duplicated doc norm stages" taggedAs FastTest in {

    val spark = SparkAccessor.spark

    val data: DataFrame =
      spark.createDataFrame(List(("Some title!", "<html>"))).toDF("title", "description")

    data.show()

    val documentAssembler = new DocumentAssembler()
      .setInputCol("description")
      .setOutputCol("document")

    val tag_remover = new DocumentNormalizer()
      .setInputCols("document")
      .setOutputCol("tag_removed")
      .setAction("clean")
      .setPatterns(Array("<[^>]*>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});"))
      .setReplacement(" ")
      .setPolicy("pretty_all")
      .setLowercase(false)
      .setEncoding("UTF-8")

    val date_remover = new DocumentNormalizer()
      .setInputCols("tag_removed")
      .setOutputCol("dates_removed")
      .setAction("clean")
      .setPatterns(Array("""(\d{1,4}[\-|\/|\.]\d{1,4}[\-|\/|\.]\d{1,4})"""
        + """|(\d{1,4}[\-|\/|\.]\d{1,4})"""
        + """|(([Jj][anuary]+|[Ff][ebruary]+|[Mm][arch]+|[Aa][pril]+|[Mm][ay]|[Jj][une]+|[Jj][uly]+|[Aa][ugust]+|[Ss][eptember]+|[Oo][ctober]+|[Nn][ovember]+|[Dd][ecember]+)\.?\s\d{1,2}\,?\s\d{2,4})"""
        + """|(\d{1,4}\,? ([Jj][anuary]+|[Ff][ebruary]+|[Mm][arch]+|[Aa][pril]+|[Mm][ay]|[Jj][une]+|[Jj][uly]+|[Aa][ugust]+|[Ss][eptember]+|[Oo][ctober]+|[Nn][ovember]+|[Dd][ecember]+)\.?\,? \d{1,4})"""))
      .setReplacement("")
      .setPolicy("pretty_all")
      .setLowercase(false)
      .setEncoding("UTF-8")

    val docPatternRemoverPipeline =
      new Pipeline().setStages(Array(documentAssembler, tag_remover, date_remover))

    println(Annotation.apply(""))
    docPatternRemoverPipeline.fit(data).transform(data).show(false)
  }
}
