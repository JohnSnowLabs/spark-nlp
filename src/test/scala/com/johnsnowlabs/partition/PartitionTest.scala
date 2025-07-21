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
package com.johnsnowlabs.partition

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.reader.{ElementType, HTMLElement}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.sql.functions.col
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable

class PartitionTest extends AnyFlatSpec {

  val txtDirectory = "src/test/resources/reader/txt"
  val wordDirectory = "src/test/resources/reader/doc"
  val excelDirectory = "src/test/resources/reader/xls"
  val powerPointDirectory = "src/test/resources/reader/ppt"
  val emailDirectory = "src/test/resources/reader/email"
  val htmlDirectory = "src/test/resources/reader/html"
  val pdfDirectory = "src/test/resources/reader/pdf"
  val xmlDirectory = "src/test/resources/reader/xml"

  "Partition" should "work with text content_type" taggedAs FastTest in {
    val textDf = Partition(Map("content_type" -> "text/plain")).partition(txtDirectory)

    assert(!textDf.select(col("txt").getItem(0)).isEmpty)
  }

  it should "identify text file" taggedAs FastTest in {
    val textDf = Partition().partition(s"$txtDirectory/simple-text.txt")

    assert(!textDf.select(col("txt").getItem(0)).isEmpty)
  }

  it should "work with word content_type" taggedAs FastTest in {
    val wordDf = Partition(Map("content_type" -> "application/msword")).partition(wordDirectory)

    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
  }

  it should "identify word file" taggedAs FastTest in {
    val wordDf = Partition().partition(s"$wordDirectory/fake_table.docx")

    assert(!wordDf.select(col("doc").getItem(0)).isEmpty)
  }

  it should "work with excel content_type" taggedAs FastTest in {
    val excelDf =
      Partition(Map("content_type" -> "application/vnd.ms-excel")).partition(excelDirectory)

    assert(!excelDf.select(col("xls").getItem(0)).isEmpty)
  }

  it should "identify excel file" taggedAs FastTest in {
    val excelDf = Partition().partition(s"$excelDirectory/vodafone.xlsx")

    assert(!excelDf.select(col("xls").getItem(0)).isEmpty)
  }

  it should "work with email content_type" taggedAs FastTest in {
    val emailDf = Partition(Map("content_type" -> "message/rfc822")).partition(emailDirectory)

    assert(!emailDf.select(col("email").getItem(0)).isEmpty)
  }

  it should "wok with email file" taggedAs FastTest in {
    val emailDf = Partition().partition(s"$emailDirectory/test-several-attachments.eml")

    assert(!emailDf.select(col("email").getItem(0)).isEmpty)
  }

  it should "work with powerpoint content_type" taggedAs FastTest in {
    val pptDf = Partition(Map("content_type" -> "application/vnd.ms-powerpoint"))
      .partition(powerPointDirectory)

    assert(!pptDf.select(col("ppt").getItem(0)).isEmpty)
  }

  it should "identify powerpoint file" taggedAs FastTest in {
    val pptDf = Partition().partition(s"$powerPointDirectory/fake-power-point.pptx")

    assert(!pptDf.select(col("ppt").getItem(0)).isEmpty)
  }

  it should "work with html content_type" taggedAs FastTest in {
    val htmlDf = Partition(Map("content_type" -> "text/html")).partition(htmlDirectory)

    assert(!htmlDf.select(col("html").getItem(0)).isEmpty)
  }

  it should "identify html file" taggedAs FastTest in {
    val htmlDf = Partition().partition(s"$htmlDirectory/fake-html.html")

    assert(!htmlDf.select(col("html").getItem(0)).isEmpty)
  }

  it should "work with an URL" taggedAs SlowTest in {
    val htmlDf = Partition().partition("https://www.wikipedia.org")

    assert(!htmlDf.select(col("html").getItem(0)).isEmpty)
  }

  it should "work with a set of URLS" taggedAs SlowTest in {
    val htmlDf =
      Partition().partitionUrls(Array("https://www.wikipedia.org", "https://example.com/"))

    assert(!htmlDf.select(col("html").getItem(0)).isEmpty)
  }

  it should "identify a PDF file" taggedAs FastTest in {
    val pdfDf = Partition().partition(s"$pdfDirectory/text_3_pages.pdf")

    assert(!pdfDf.select(col("pdf")).isEmpty)
  }

  it should "work with PDF content_type" taggedAs FastTest in {
    val pdfDf = Partition(Map("content_type" -> "application/pdf")).partition(pdfDirectory)

    assert(!pdfDf.select(col("pdf")).isEmpty)
  }

  it should "work with text in memory" taggedAs FastTest in {
    import ResourceHelper.spark.implicits._
    val content =
      """
        |The big brown fox
        |was walking down the lane.
        |
        |At the end of the lane,
        |the fox met a bear.
        |""".stripMargin

    val textDf = Partition(Map("groupBrokenParagraphs" -> "true")).partitionText(content)

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

    assert(elements.head.elementType == expectedElements.head.elementType)
    assert(elements.head.content == expectedElements.head.content)
  }

  it should "work with XML content_type" taggedAs FastTest in {
    val pdfDf = Partition(Map("content_type" -> "application/xml")).partition(xmlDirectory)

    assert(!pdfDf.select(col("xml")).isEmpty)
  }

}
