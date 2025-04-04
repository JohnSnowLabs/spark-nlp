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
import org.apache.spark.sql.functions.col
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

    assert(elements == expectedElements)
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

    assert(elements == expectedElements)
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

    assert(elements == expectedElements)
  }

}
