package com.johnsnowlabs.reader.util

import org.apache.poi.ss.usermodel.{Cell, CellType, DateUtil, HorizontalAlignment, Row}

import scala.collection.JavaConverters._

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

}
