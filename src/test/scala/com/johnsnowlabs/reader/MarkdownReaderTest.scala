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
package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source

class MarkdownReaderTest extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._
  val mdDirectory = "src/test/resources/reader/md"
  val mdReader = new MarkdownReader()

  "Markdown Reader" should "read a markdown file with headers and text" taggedAs FastTest in {
    val textDf = mdReader.md(s"$mdDirectory/simple.md")
    textDf.select("md").show(truncate = false)

    val elements: Seq[HTMLElement] = textDf
      .select("md")
      .as[Seq[HTMLElement]]
      .collect()
      .head

    assert(elements.exists(_.elementType == ElementType.TITLE))
    assert(elements.exists(_.elementType == ElementType.NARRATIVE_TEXT))
  }

  it should "detect list items in markdown" taggedAs FastTest in {
    val content = """
                    |# Shopping List
                    | - Milk
                    | - Bread
                    | - Eggs
                    |""".stripMargin

    val mdDf = mdReader.md(text = content)
    val elements: Seq[HTMLElement] = mdDf
      .select("md")
      .as[Seq[HTMLElement]]
      .collect()
      .head

    assert(elements.count(_.elementType == ElementType.LIST_ITEM) == 3)
    assert(elements.exists(e => e.content.contains("Bread")))
  }

  it should "parse code blocks in markdown" taggedAs FastTest in {
    val content = """
                    |```scala
                    |val x = 10
                    |println(x)
                    |```
                    |""".stripMargin

    val mdDf = mdReader.md(text = content)
    val elements: Seq[HTMLElement] = mdDf
      .select("md")
      .as[Seq[HTMLElement]]
      .collect()
      .head

    assert(elements.exists(_.elementType == ElementType.UNCATEGORIZED_TEXT))
    assert(elements.exists(_.content.contains("val x = 10")))
  }

  it should "parse README.md and the first element must be a TITLE" taggedAs FastTest in {
    val mdDf = mdReader.md(filePath = s"$mdDirectory/README.md")

    val elements: Seq[HTMLElement] = mdDf
      .select(mdReader.getOutputColumn)
      .as[Seq[HTMLElement]]
      .collect()
      .head

    assert(elements.nonEmpty, "No elements found in README.md")
    assert(
      elements.head.elementType == ElementType.TITLE,
      s"First element type is not TITLE, but ${elements.head.elementType}")

  }

  it should "parse README.md from direct text input" taggedAs FastTest in {
    val source = Source.fromFile(s"$mdDirectory/README.md", "UTF-8")
    val content =
      try source.mkString
      finally source.close()
    val df = mdReader.md(text = content)

    val elements: Seq[HTMLElement] = df
      .select(mdReader.getOutputColumn)
      .as[Seq[HTMLElement]]
      .collect()
      .head

    assert(elements.nonEmpty, "No elements found from text input")
  }

  it should "parse markdown from a real GitHub raw URL" taggedAs SlowTest in {
    val testUrl =
      "https://raw.githubusercontent.com/adamschwartz/github-markdown-kitchen-sink/master/README.md"
    val mdDf = mdReader.md(url = testUrl)
    val elements: Seq[HTMLElement] = mdDf
      .select(mdReader.getOutputColumn)
      .as[Seq[HTMLElement]]
      .collect()
      .head

    mdDf.show()
    assert(elements.nonEmpty, "Parsed elements from URL are empty")

    val sourceVal = mdDf.select("source").as[String].collect().head
    assert(sourceVal == testUrl, s"Source column mismatch: expected $testUrl, got $sourceVal")
  }

  it should "parse markdown table as TABLE element" taggedAs FastTest in {
    val df = mdReader.md(filePath = s"$mdDirectory/simple-table.md")
    val elements: Seq[HTMLElement] = df
      .select(mdReader.getOutputColumn)
      .as[Seq[HTMLElement]]
      .collect()
      .head

    assert(elements.nonEmpty, "Parsed elements for table are empty")
    assert(elements.head.elementType == ElementType.TABLE)
  }

  it should "correctly parse markdown files with umlauts regardless of encoding" taggedAs FastTest in {
    val filesAndEncodings =
      Seq(("umlauts-utf8.md", "UTF-8"), ("umlauts-non-utf8.md", "ISO-8859-1"))

    filesAndEncodings.foreach { case (fname, enc) =>
      val path = s"$mdDirectory/$fname"
      val mdReader = new MarkdownReader(fileEncoding = enc)
      val df = mdReader.md(filePath = path)
      val elements: Seq[HTMLElement] = df
        .select(mdReader.getOutputColumn)
        .as[Seq[HTMLElement]]
        .collect()
        .head
      assert(elements.nonEmpty, s"File $fname: No elements parsed")
      assert(
        elements.last.content.endsWith("äöüß"),
        s"File $fname: Last element does not end with 'äöüß', got: ${elements.last.content}")
    }
  }

  it should "parse a code block with XML processing instruction as a single element" taggedAs FastTest in {
    val xmlContent =
      """```
        |<?xml version="1.0"?>
        |<sparql xmlns="http://www.w3.org/2005/sparql-results#">
        |  <head></head>
        |  <boolean>true</boolean>
        |</sparql>
        |```""".stripMargin

    val mdDf = mdReader.md(text = xmlContent)
    val elements: Seq[HTMLElement] = mdDf
      .select(mdReader.getOutputColumn)
      .as[Seq[HTMLElement]]
      .collect()
      .head

    assert(elements.length == 1, s"Expected 1 element, got ${elements.length}")
  }

  it should "parse a code block with no XML processing instruction" taggedAs FastTest in {
    val xmlContent =
      """```
        |<?php echo "hello"; ?>
        |```""".stripMargin

    val mdDf = mdReader.md(text = xmlContent)
    val elements: Seq[HTMLElement] = mdDf
      .select(mdReader.getOutputColumn)
      .as[Seq[HTMLElement]]
      .collect()
      .head

    assert(elements.length == 1, s"Expected 1 element, got ${elements.length}")
  }

  it should "parse markdown table as HTML element" taggedAs FastTest in {
    val mdReader = new MarkdownReader(outputFormat = "html-table")
    val mdDf = mdReader.md(filePath = s"$mdDirectory/simple-table.md")
    val elements: Seq[HTMLElement] = mdDf
      .select(mdReader.getOutputColumn)
      .as[Seq[HTMLElement]]
      .collect()
      .head

    assert(elements.nonEmpty, "Parsed elements for table are empty")
    assert(elements.head.elementType == ElementType.TABLE)
    assert(elements.head.content.contains("<table>"), "Table HTML content is missing <table> tag")
  }

  it should "parse markdown table as JSON element" taggedAs FastTest in {
    val mdReader = new MarkdownReader(outputFormat = "json-table")
    val mdDf = mdReader.md(filePath = s"$mdDirectory/simple-table.md")
    val elements: Seq[HTMLElement] = mdDf
      .select(mdReader.getOutputColumn)
      .as[Seq[HTMLElement]]
      .collect()
      .head

    assert(elements.nonEmpty, "Parsed elements for table are empty")
    assert(elements.head.elementType == ElementType.TABLE)
    assert(elements.head.content.contains("header"), "JSON content is missing 'header' key")
  }

}
