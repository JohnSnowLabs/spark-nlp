package com.johnsnowlabs.reader.util

import com.johnsnowlabs.reader.{ElementType, HTMLElement}
import org.apache.poi.ss.usermodel.{Cell, CellType, DateUtil, HorizontalAlignment, Row, Sheet}

import scala.collection.JavaConverters._
import scala.collection.mutable

object XlsxParser {

  implicit class RichRow(row: Row) {

    def isTitle(titleFontSizeThreshold: Int): Boolean = {
      row.cellIterator().asScala.exists { cell =>
        val cellStyle = cell.getCellStyle
        val font = row.getSheet.getWorkbook.getFontAt(cellStyle.getFontIndexAsInt)

        val isBold = font.getBold
        val isCentered = cellStyle.getAlignment == HorizontalAlignment.CENTER

        val text = cell.getCellValue.trim
        val isUppercaseOrCapitalized =
          text.nonEmpty && (text == text.toUpperCase || text.headOption.exists(_.isUpper))

        val fontSize = font.getFontHeightInPoints
        val isLargeFont = fontSize >= titleFontSizeThreshold

        (isBold && isCentered) || (isBold && isUppercaseOrCapitalized) || (isBold && isLargeFont)
      }
    }
  }

  implicit class RichCell(cell: Cell) {

    def getCellValue: String = {
      cell.getCellType match {
        case CellType.STRING => cell.getStringCellValue
        case CellType.NUMERIC =>
          if (DateUtil.isCellDateFormatted(cell))
            cell.getDateCellValue.toString
          else
            cell.getNumericCellValue.toString
        case CellType.BOOLEAN => cell.getBooleanCellValue.toString
        case CellType.FORMULA => cell.getCellFormula
        case _ => ""
      }
    }

  }

  implicit class RichSheet(sheet: Sheet) {

    def buildHtmlIfNeeded(
        elementsBuffer: mutable.ArrayBuffer[HTMLElement],
        outputFormat: String): Unit = {
      val rowsHtml = sheet
        .iterator()
        .asScala
        .flatMap { row =>
          val cellsHtml = row
            .cellIterator()
            .asScala
            .flatMap { cell =>
              val cellValue = cell.getCellValue.trim
              if (cellValue.nonEmpty) Some(s"<td>$cellValue</td>") else None
            }
            .mkString("")
          if (cellsHtml.nonEmpty) Some(s"<tr>$cellsHtml</tr>") else None
        }
        .mkString("")

      val sheetHtml = if (rowsHtml.nonEmpty) s"<table>$rowsHtml</table>" else ""
      if (sheetHtml.nonEmpty) {
        if (outputFormat == "html-table") {
          val htmlElement =
            HTMLElement(
              ElementType.HTML,
              sheetHtml,
              mutable.Map("SheetName" -> sheet.getSheetName))
          elementsBuffer += htmlElement
        } else if (outputFormat == "json-table") {
          val tableElement = HTMLParser.parseFirstTableElement(sheetHtml)
          val jsonString = HTMLParser.tableElementToJson(tableElement)
          val jsonElement =
            HTMLElement(
              ElementType.JSON,
              jsonString,
              mutable.Map("SheetName" -> sheet.getSheetName))
          elementsBuffer += jsonElement
        }
      }
    }
  }

}
