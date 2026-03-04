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
package com.johnsnowlabs.reader

import com.johnsnowlabs.reader.util.AssertReaders
import com.johnsnowlabs.tags.FastTest
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.PDPageContentStream
import org.apache.pdfbox.pdmodel.common.PDRectangle
import org.apache.pdfbox.pdmodel.font.PDType1Font
import org.apache.pdfbox.pdmodel.graphics.image.LosslessFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.apache.spark.sql.functions.{col, explode}

import java.awt.Color
import java.awt.image.BufferedImage
import java.nio.file.Files

class PdfReaderTest extends AnyFlatSpec {

  val pdfDirectory = "src/test/resources/reader/pdf/"

  "PdfReader" should "read a PDF file as dataframe" taggedAs FastTest in {
    val pdfReader = new PdfReader()
    val pdfDf = pdfReader.pdf(s"$pdfDirectory/text_3_pages.pdf")

    assert(!pdfDf.select(col("pdf").getItem(0)).isEmpty)
    assert(!pdfDf.columns.contains("content"))
  }

  it should "store content" taggedAs FastTest in {
    val pdfReader = new PdfReader(storeContent = true)
    val pdfDf = pdfReader.pdf(s"$pdfDirectory/text_3_pages.pdf")

    assert(!pdfDf.select(col("pdf").getItem(0)).isEmpty)
    assert(pdfDf.columns.contains("content"))
  }

  it should "include coord field in IMAGE metadata with {x:...,y:...} format when readAsImage is enabled" taggedAs FastTest in {
    val pdfReader = new PdfReader(readAsImage = true)
    val pdfDf = pdfReader.pdf(s"$pdfDirectory/text_3_pages.pdf")

    val imagesDf = pdfDf
      .select(explode(col("pdf")).as("pdf_exploded"))
      .filter(col("pdf_exploded.elementType") === ElementType.IMAGE)

    val coordDf = imagesDf.selectExpr("pdf_exploded.metadata.coord as coord")
    val pageDf =
      imagesDf.selectExpr("cast(pdf_exploded.metadata.pageNumber as int) as pageNumber")

    assert(coordDf.count() > 0, "No IMAGE elements found in PdfReader output")
    assert(coordDf.filter(col("coord").isNull).count() == 0, "Missing coord in IMAGE metadata")
    assert(
      pageDf.filter(col("pageNumber").isNull).count() == 0,
      "Missing pageNumber in IMAGE metadata")
    assert(
      pageDf.filter(col("pageNumber") < 1).count() == 0,
      "IMAGE pageNumber should be 1-based")

    val pattern = """\{x:\d+,y:\d+\}"""
    val allMatch = coordDf.collect().forall(row => row.getAs[String]("coord").matches(pattern))
    assert(allMatch, "Some IMAGE coord fields do not match expected {x:...,y:...} format")
  }

  it should "identify text as titles based on threshold value" taggedAs FastTest in {
    val pdfReader = new PdfReader(titleThreshold = 10)
    val pdfDf = pdfReader.pdf(s"$pdfDirectory/pdf-title.pdf")

    val titleDF = pdfDf
      .select(explode(col("pdf")).as("exploded_pdf"))
      .filter(col("exploded_pdf.elementType") === ElementType.TITLE)

    assert(titleDF.count() == 3)
  }

  it should "include paragraph_index and paragraph_y metadata fields for text elements" taggedAs FastTest in {
    val paragraphSpacingY = 25
    val pdfReader = new PdfReader()
    val pdfDf = pdfReader.pdf(s"$pdfDirectory/text_3_pages.pdf")

    val textDf = pdfDf
      .select(explode(col("pdf")).as("pdf_exploded"))
      .filter(col("pdf_exploded.elementType")
        .isin(ElementType.TITLE, ElementType.NARRATIVE_TEXT))
      .selectExpr(
        "cast(pdf_exploded.metadata.paragraph_index as int) as paragraphIndex",
        "cast(pdf_exploded.metadata.paragraph_y as int) as paragraphY")

    assert(textDf.count() > 0, "No text elements found in PdfReader output")
    assert(
      textDf.filter(col("paragraphIndex").isNull).count() == 0,
      "Missing paragraph_index in text metadata")
    assert(
      textDf.filter(col("paragraphY").isNull).count() == 0,
      "Missing paragraph_y in text metadata")
    assert(
      textDf.filter(col("paragraphY") =!= col("paragraphIndex") * paragraphSpacingY).count() == 0,
      "paragraph_y should be derived from paragraph_index")
  }

  it should "extract non-zero embedded image coordinates when PDF contains positioned images" taggedAs FastTest in {
    val tempPdfPath = Files.createTempFile("pdf-reader-image-coord-", ".pdf")
    val document = new PDDocument()

    try {
      val page = new PDPage(PDRectangle.LETTER)
      document.addPage(page)

      val image = new BufferedImage(120, 60, BufferedImage.TYPE_INT_RGB)
      val graphics = image.createGraphics()
      graphics.setColor(Color.WHITE)
      graphics.fillRect(0, 0, image.getWidth, image.getHeight)
      graphics.setColor(Color.BLUE)
      graphics.fillRect(0, 0, image.getWidth, image.getHeight)
      graphics.dispose()

      val pdImage = LosslessFactory.createFromImage(document, image)
      val contentStream = new PDPageContentStream(document, page)
      contentStream.beginText()
      contentStream.setFont(PDType1Font.HELVETICA_BOLD, 14)
      contentStream.newLineAtOffset(72, 740)
      contentStream.showText("Revenue Summary")
      contentStream.endText()
      contentStream.drawImage(pdImage, 100, 320, 220, 120)
      contentStream.close()

      document.save(tempPdfPath.toFile)
    } finally {
      document.close()
    }

    try {
      val pdfReader = new PdfReader(readAsImage = true)
      val pdfDf = pdfReader.pdf(tempPdfPath.toString)

      val imagesDf = pdfDf
        .select(explode(col("pdf")).as("pdf_exploded"))
        .filter(col("pdf_exploded.elementType") === ElementType.IMAGE)

      val coordPattern = """\{x:(\d+),y:(\d+)\}""".r
      val yValues = imagesDf
        .selectExpr("pdf_exploded.metadata.coord as coord")
        .collect()
        .flatMap(row =>
          coordPattern
            .findFirstMatchIn(row.getAs[String]("coord"))
            .map(_.group(2).toInt))

      assert(yValues.nonEmpty, "Expected IMAGE elements with coord metadata")
      assert(yValues.exists(_ > 0), "Expected at least one embedded IMAGE with y > 0")
    } finally {
      Files.deleteIfExists(tempPdfPath)
    }
  }

  it should "handle corrupted files" taggedAs FastTest in {
    val pdfReader = new PdfReader()
    val pdfDf = pdfReader.pdf(s"src/test/resources/reader/pdf-corrupted/corrupted.pdf")

    val resultDF = pdfDf
      .select(explode(col("pdf")).as("exploded_pdf"))
      .filter(col("exploded_pdf.elementType") === ElementType.ERROR)

    assert(resultDF.count() == 1)
  }

  it should "output hierarchy metadata" in {
    val pdfReader = new PdfReader()
    val pdfDf = pdfReader.pdf(s"$pdfDirectory/hierarchy_test.pdf")

    AssertReaders.assertHierarchy(pdfDf, "pdf")
  }

}
