/*
 * Copyright 2017-2024 John Snow Labs
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
import com.johnsnowlabs.reader.util.XlsxParser.{RichCell, RichRow, RichSheet}
import org.apache.poi.hssf.usermodel.{HSSFSheet, HSSFWorkbook}
import org.apache.poi.ss.usermodel.Workbook
import org.apache.poi.xssf.usermodel.{XSSFSheet, XSSFWorkbook}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import java.io.ByteArrayInputStream
import scala.collection.JavaConverters._
import scala.collection.mutable

class ExcelReader(
    titleFontSize: Int = 9,
    cellSeparator: String = "\t",
    storeContent: Boolean = false,
    includePageBreaks: Boolean = false,
    inferTableStructure: Boolean = false,
    appendCells: Boolean = false)
    extends Serializable {

  private val spark = ResourceHelper.spark
  import spark.implicits._

  def xls(filePath: String): DataFrame = {
    if (ResourceHelper.validFile(filePath)) {
      val binaryFilesRDD = spark.sparkContext.binaryFiles(filePath)
      val byteArrayRDD = binaryFilesRDD.map { case (path, portableDataStream) =>
        val byteArray = portableDataStream.toArray()
        (path, byteArray)
      }
      val excelDf = byteArrayRDD
        .toDF("path", "content")
        .withColumn("xls", parseExcelUDF(col("content")))
      if (storeContent) excelDf.select("path", "xls", "content")
      else excelDf.select("path", "xls")
    } else throw new IllegalArgumentException(s"Invalid filePath: $filePath")
  }

  private val parseExcelUDF = udf((data: Array[Byte]) => {
    parseExcel(data)
  })

  // Constants for file type identification
  private val ZipMagicNumberFirstByte: Byte = 0x50.toByte // First byte of ZIP files
  private val ZipMagicNumberSecondByte: Byte = 0x4b.toByte // Second byte of ZIP files
  private val OleMagicNumber: Array[Byte] =
    Array(0xd0.toByte, 0xcf.toByte, 0x11.toByte, 0xe0.toByte) // OLE file header

  private def isXlsxFile(content: Array[Byte]): Boolean = {
    content.length > 1 &&
    content(0) == ZipMagicNumberFirstByte &&
    content(1) == ZipMagicNumberSecondByte
  }

  private def isXlsFile(content: Array[Byte]): Boolean = {
    content.length >= 4 && content.slice(0, 4).sameElements(OleMagicNumber)
  }

  private def parseExcel(content: Array[Byte]): Seq[HTMLElement] = {
    val workbookInputStream = new ByteArrayInputStream(content)
    val workbook: Workbook =
      if (isXlsxFile(content)) new XSSFWorkbook(workbookInputStream)
      else if (isXlsFile(content)) new HSSFWorkbook(workbookInputStream)
      else throw new IllegalArgumentException("Unsupported file format: must be .xls or .xlsx")

    val elementsBuffer = mutable.ArrayBuffer[HTMLElement]()

    for (sheetIndex <- 0 until workbook.getNumberOfSheets) {
      if (includePageBreaks)
        buildSheetContentWithPageBreaks(workbook, sheetIndex, elementsBuffer)
      else
        buildSheetContent(workbook, sheetIndex, elementsBuffer)
    }

    workbook.close()
    elementsBuffer
  }

  private def buildSheetContent(
      workbook: Workbook,
      sheetIndex: Int,
      elementsBuffer: mutable.ArrayBuffer[HTMLElement]): Unit = {

    val sheet = workbook.getSheetAt(sheetIndex)
    val sheetName = sheet.getSheetName

    val rowIterator = sheet.iterator()

    val allContents = new StringBuilder
    val allMetadata = mutable.Map[String, String]("SheetName" -> sheetName)

    while (rowIterator.hasNext) {
      val row = rowIterator.next()
      val rowIndex = row.getRowNum

      val elementType =
        if (row.isTitle(titleFontSize)) ElementType.TITLE else ElementType.NARRATIVE_TEXT

      val cellValuesWithMetadata = row
        .cellIterator()
        .asScala
        .map { cell =>
          val cellIndex = cell.getColumnIndex
          val cellValue = cell.getCellValue.trim

          val cellMetadata = mutable.Map(
            "SheetName" -> sheetName,
            "location" -> s"(${rowIndex.toString}, ${cellIndex.toString})")

          (cellValue, cellMetadata)
        }
        .toSeq

      val content = cellValuesWithMetadata.map(_._1).mkString(cellSeparator).trim

      if (content.nonEmpty) {
        if (appendCells) {
          if (allContents.nonEmpty) allContents.append("\n")
          allContents.append(content)
        } else {
          val rowMetadata = cellValuesWithMetadata
            .flatMap(_._2)
            .toMap

          val element = HTMLElement(
            elementType = elementType,
            content = content,
            metadata = mutable.Map(rowMetadata.toSeq: _*))
          elementsBuffer += element
        }
      }
    }

    if (appendCells && allContents.nonEmpty) {
      elementsBuffer += HTMLElement(
        elementType = ElementType.NARRATIVE_TEXT,
        content = allContents.toString(),
        metadata = allMetadata)
    }

    if (inferTableStructure) sheet.buildHtmlIfNeeded(elementsBuffer)
  }

  private def buildSheetContentWithPageBreaks(
      workbook: Workbook,
      sheetIndex: Int,
      elementsBuffer: mutable.ArrayBuffer[HTMLElement]): Unit = {
    val sheet = workbook.getSheetAt(sheetIndex)
    val sheetName = sheet.getSheetName

    val colBreaks: Seq[Int] = sheet match {
      case xssf: XSSFSheet =>
        if (xssf.getCTWorksheet.isSetColBreaks)
          xssf.getCTWorksheet.getColBreaks.getBrkList.asScala.map(_.getId.toInt).sorted
        else Seq.empty[Int]
      case hssf: HSSFSheet =>
        Option(hssf.getColumnBreaks).map(_.toSeq).getOrElse(Seq.empty[Int])
      case _ => Seq.empty[Int]
    }

    val rowIterator = sheet.iterator()
    while (rowIterator.hasNext) {
      val row = rowIterator.next()
      val rowIndex = row.getRowNum

      val elementType =
        if (row.isTitle(titleFontSize)) ElementType.TITLE else ElementType.NARRATIVE_TEXT

      val cellsByPage: Map[Int, Seq[org.apache.poi.ss.usermodel.Cell]] =
        row
          .cellIterator()
          .asScala
          .toSeq
          .groupBy(cell => getPageNumberForCell(cell.getColumnIndex, colBreaks))

      for ((page, cells) <- cellsByPage) {
        val cellValuesWithMetadata = cells.map { cell =>
          val cellIndex = cell.getColumnIndex
          val cellValue = cell.getCellValue.trim
          val cellMetadata =
            mutable.Map("location" -> s"($rowIndex, $cellIndex)", "SheetName" -> sheetName)
          (cellValue, cellMetadata)
        }
        val content = cellValuesWithMetadata.map(_._1).mkString(cellSeparator).trim

        if (content.nonEmpty) {
          val rowMetadata = cellValuesWithMetadata.flatMap(_._2).toMap
          val elementMetadata = mutable.Map(rowMetadata.toSeq: _*)
          elementMetadata += ("pageBreak" -> page.toString)
          val element =
            HTMLElement(elementType = elementType, content = content, metadata = elementMetadata)
          elementsBuffer += element
        }
      }
    }
    if (inferTableStructure) sheet.buildHtmlIfNeeded(elementsBuffer)
  }

  private def getPageNumberForCell(cellIndex: Int, breaks: Seq[Int]): Int = {
    breaks.count(break => cellIndex > break) + 1
  }

}
