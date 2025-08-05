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

import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.functions.{col, explode}
import org.scalatest.flatspec.AnyFlatSpec
import com.johnsnowlabs.nlp.util.io.ResourceHelper

import scala.collection.mutable

class TextReaderTest extends AnyFlatSpec {

  val txtDirectory = "src/test/resources/reader/txt/"

  "Text Reader" should "read a directory of text files" taggedAs FastTest in {
    val textReader = new TextReader()
    val textDf = textReader.txt(s"$txtDirectory/simple-text.txt")
    textDf.select("txt").show(false)

    assert(!textDf.select(col("txt").getItem(0)).isEmpty)
    assert(!textDf.columns.contains("content"))
  }

  "Text Reader" should "store content" taggedAs FastTest in {
    val textReader = new TextReader(storeContent = true)
    val textDf = textReader.txt(txtDirectory)
    textDf.show()

    assert(!textDf.select(col("txt").getItem(0)).isEmpty)
    assert(textDf.columns.contains("content"))
  }

  it should "group broken paragraphs" taggedAs FastTest in {
    import ResourceHelper.spark.implicits._

    val textReader = new TextReader(groupBrokenParagraphs = true)
    val content =
      """
        |The big brown fox
        |was walking down the lane.
        |
        |At the end of the lane,
        |the fox met a bear.
        |""".stripMargin
    val textDf = textReader.txtContent(content)
    textDf.show(truncate = false)

    val elements: Seq[HTMLElement] = textDf
      .select("txt")
      .as[Seq[HTMLElement]]
      .collect()
      .head

    val expectedElements = Seq(
      HTMLElement(
        ElementType.NARRATIVE_TEXT,
        "The big brown fox was walking down the lane.",
        mutable.Map("paragraph" -> "0")),
      HTMLElement(
        ElementType.NARRATIVE_TEXT,
        "At the end of the lane, the fox met a bear.",
        mutable.Map("paragraph" -> "0")))

    val actualBasic = elements.map(e => (e.elementType, e.content))
    val expectedBasic = expectedElements.map(e => (e.elementType, e.content))
    assert(actualBasic == expectedBasic)
  }

  it should "group broken paragraphs reading from file" taggedAs FastTest in {
    import ResourceHelper.spark.implicits._
    val textReader = new TextReader(groupBrokenParagraphs = true)
    val textDf = textReader.txt(s"$txtDirectory/test-paragraph.txt")
    textDf.show(truncate = false)

    val elements: Seq[HTMLElement] = textDf
      .select("txt")
      .as[Seq[HTMLElement]]
      .collect()
      .head

    val expectedElements = Seq(
      HTMLElement(
        ElementType.NARRATIVE_TEXT,
        "The big brown fox was walking down the lane.",
        mutable.Map("paragraph" -> "0")),
      HTMLElement(
        ElementType.NARRATIVE_TEXT,
        "At the end of the lane, the fox met a bear.",
        mutable.Map("paragraph" -> "0")))

    val actualBasic = elements.map(e => (e.elementType, e.content))
    val expectedBasic = expectedElements.map(e => (e.elementType, e.content))
    assert(actualBasic == expectedBasic)
  }

  it should "paragraph split with custom regex" taggedAs FastTest in {
    import ResourceHelper.spark.implicits._
    val textReader =
      new TextReader(groupBrokenParagraphs = true, paragraphSplit = """(\s*\n\s*){3}""")
    val content = """The big red fox

is walking down the lane.


At the end of the lane

the fox met a friendly bear."""
    val textDf = textReader.txtContent(content)
    textDf.show(truncate = false)

    val elements: Seq[HTMLElement] = textDf
      .select("txt")
      .as[Seq[HTMLElement]]
      .collect()
      .head

    val expectedElements = Seq(
      HTMLElement(
        ElementType.NARRATIVE_TEXT,
        "The big red fox is walking down the lane.",
        mutable.Map("paragraph" -> "0")),
      HTMLElement(
        ElementType.NARRATIVE_TEXT,
        "At the end of the lane the fox met a friendly bear.",
        mutable.Map("paragraph" -> "0")))

    val actualBasic = elements.map(e => (e.elementType, e.content))
    val expectedBasic = expectedElements.map(e => (e.elementType, e.content))
    assert(actualBasic == expectedBasic)
  }

  it should "output as title for font size >= 40" taggedAs FastTest in {
    val textReader = new TextReader(blockSplit = "\\n", titleLengthSize = 40)

    val textDf = textReader.txt(s"$txtDirectory/title-length-test.txt")

    val titleDF = textDf
      .select(explode(col("txt")).as("exploded_txt"))
      .filter(col("exploded_txt.elementType") === ElementType.TITLE)
    titleDF.select("exploded_txt").show(truncate = false)

    assert(titleDF.count() == 1)
  }

  it should "output as title for font size >= 80" taggedAs FastTest in {
    val textReader = new TextReader(blockSplit = "\\n", titleLengthSize = 80)

    val textDf = textReader.txt(s"$txtDirectory/title-length-test.txt")

    val titleDF = textDf
      .select(explode(col("txt")).as("exploded_txt"))
      .filter(col("exploded_txt.elementType") === ElementType.TITLE)
    titleDF.select("exploded_txt").show(truncate = false)

    assert(titleDF.count() == 2)
  }

  it should "merge lines with more than 5 words into a single paragraph block" taggedAs FastTest in {
    val textReader = new TextReader(groupBrokenParagraphs = true, shortLineWordThreshold = 5)

    val textDf = textReader.txt(s"$txtDirectory/short-line-test.txt")

    val explodedDf = textDf.select(explode(col("txt")).as("exploded_txt"))
    explodedDf.select("exploded_txt").show(truncate = false)

    assert(explodedDf.count() == 4)
  }

  it should "treat all lines with fewer than 10 words as individual paragraph blocks" taggedAs FastTest in {
    val textReader = new TextReader(groupBrokenParagraphs = true, shortLineWordThreshold = 10)

    val textDf = textReader.txt(s"$txtDirectory/short-line-test.txt")

    val explodedDf = textDf.select(explode(col("txt")).as("exploded_txt"))
    explodedDf.select("exploded_txt").show(truncate = false)

    assert(explodedDf.count() == 5)
  }

  it should "trigger line-based splitting when the empty line ratio is below the threshold" taggedAs FastTest in {
    val reader = new TextReader(
      groupBrokenParagraphs = true,
      threshold = 0.5 // High threshold → encourages line splitting
    )

    val textDf = reader.txt(s"$txtDirectory/threshold-test.txt")
    val explodedDf = textDf.select(explode(col("txt")).as("exploded_txt"))
    explodedDf.select("exploded_txt").show(truncate = false)

    assert(explodedDf.count() == 5) // Each line becomes its own paragraph
  }

  it should "trigger paragraph grouping when the empty line ratio is above the threshold, merging long lines" taggedAs FastTest in {
    val reader = new TextReader(
      groupBrokenParagraphs = true,
      threshold = 0.1 // Low threshold → triggers grouping logic
    )

    val textDf = reader.txt(s"$txtDirectory/threshold-test.txt")
    val explodedDf = textDf.select(explode(col("txt")).as("exploded_txt"))
    explodedDf.select("exploded_txt").show(truncate = false)

    assert(explodedDf.count() < 5) // Merged lines reduce total paragraph count
  }

  it should "trigger line-splitting when maxLineCount restricts the ratio below threshold" taggedAs FastTest in {
    val reader = new TextReader(
      groupBrokenParagraphs = true,
      threshold = 0.3,
      maxLineCount = 5 // fewer lines analyzed → lower empty line ratio
    )

    val textDf = reader.txt(s"$txtDirectory/max-line-count-test.txt")
    val explodedDf = textDf.select(explode(col("txt")).as("exploded_txt"))
    explodedDf.select("exploded_txt").show(truncate = false)

    assert(explodedDf.count() == 6) // All lines are split individually
  }

  it should "trigger paragraph grouping when more lines are counted and ratio exceeds threshold" taggedAs FastTest in {
    val reader = new TextReader(
      groupBrokenParagraphs = true,
      threshold = 0.2,
      maxLineCount = 15 // all lines counted → higher ratio
    )

    val textDf = reader.txt(s"$txtDirectory/max-line-count-test.txt")
    val explodedDf = textDf.select(explode(col("txt")).as("exploded_txt"))
    explodedDf.select("exploded_txt").show(truncate = false)

    assert(explodedDf.count() < 6) // Paragraphs are grouped
  }

}
