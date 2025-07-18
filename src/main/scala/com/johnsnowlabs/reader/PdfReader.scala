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
import com.johnsnowlabs.partition.util.PartitionHelper.datasetWithBinaryFile
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.text.{PDFTextStripper, TextPosition}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}
import java.io.ByteArrayInputStream
import scala.collection.JavaConverters._
import scala.collection.mutable

class PdfReader(storeContent: Boolean = false, titleThreshold: Double = 18.0)
    extends Serializable {

  private lazy val spark = ResourceHelper.spark
  private var outputColumn = "pdf"

  def setOutputColumn(name: String): this.type = {
    require(name.nonEmpty, "Output column name cannot be empty.")
    outputColumn = name
    this
  }
  def getOutputColumn: String = outputColumn

  def pdf(filePath: String): DataFrame = {
    if (!ResourceHelper.validFile(filePath))
      throw new IllegalArgumentException(s"Invalid filePath: $filePath")

    val binaryDF = datasetWithBinaryFile(spark, filePath)
    val withElements = binaryDF.withColumn(outputColumn, parsePdfUDF(col("content")))
    if (storeContent) withElements.select("path", outputColumn, "content")
    else withElements.select("path", outputColumn)
  }

  private val parsePdfUDF = udf((data: Array[Byte]) => pdfToHTMLElement(data))

  def pdfToHTMLElement(content: Array[Byte]): Seq[HTMLElement] = {
    val docInputStream = new ByteArrayInputStream(content)
    try {
      val pdfDoc = PDDocument.load(docInputStream)
      val elements = extractElementsFromPdf(pdfDoc)
      pdfDoc.close()
      elements
    } catch {
      case e: Exception =>
        Seq(
          HTMLElement(
            ElementType.UNCATEGORIZED_TEXT,
            s"Could not parse PDF: ${e.getMessage}",
            mutable.Map()))
    } finally {
      docInputStream.close()
    }
  }

  private def extractElementsFromPdf(pdfDoc: PDDocument): Seq[HTMLElement] = {
    val collectedElements = mutable.ListBuffer[HTMLElement]()
    val textStripper = new PDFTextStripper() {
      override def writeString(
          text: String,
          textPositions: java.util.List[TextPosition]): Unit = {
        val lineGroups = groupTextPositionsByLine(textPositions)
        val lineElements = lineGroups.flatMap { case (_, linePositions) =>
          classifyLineElement(linePositions, getCurrentPageNo)
        }
        collectedElements ++= lineElements
      }
    }
    textStripper.setSortByPosition(true)
    textStripper.setStartPage(1)
    textStripper.setEndPage(pdfDoc.getNumberOfPages)
    textStripper.getText(pdfDoc)
    collectedElements
  }

  private def groupTextPositionsByLine(
      textPositions: java.util.List[TextPosition]): Map[Int, Seq[TextPosition]] = {
    val yTolerance = 2f // Potential parameter, since needs to experiment to fit your PDFs
    textPositions.asScala.groupBy(tp => (tp.getY / yTolerance).round)
  }

  private def classifyLineElement(
      linePositions: Seq[TextPosition],
      pageNumber: Int): Option[HTMLElement] = {
    val lineText = linePositions.map(_.getUnicode).mkString.trim
    if (lineText.isEmpty) return None

    val averageFontSize = linePositions.map(_.getFontSize).sum / linePositions.size
    val mostCommonFontName = linePositions.groupBy(_.getFont.getName).maxBy(_._2.size)._1

    val elementType =
      if (isTitle(averageFontSize, mostCommonFontName)) ElementType.TITLE
      else ElementType.NARRATIVE_TEXT

    val metadata = mutable.Map("pageNumber" -> pageNumber.toString)
    Some(HTMLElement(elementType, lineText, metadata))
  }

  private def isTitle(fontSize: Double, fontName: String): Boolean = {
    fontSize >= titleThreshold || fontName.toLowerCase.contains("bold")
  }

}
