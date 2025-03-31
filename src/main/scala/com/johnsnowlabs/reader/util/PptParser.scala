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

package com.johnsnowlabs.reader.util

import com.johnsnowlabs.reader.{ElementType, HTMLElement}
import org.apache.poi.hslf.usermodel.{HSLFSlide, HSLFTable, HSLFTextShape}
import org.apache.poi.xslf.usermodel.{XSLFSlide, XSLFTable, XSLFTextShape}

import scala.collection.JavaConverters._
import scala.collection.mutable

object PptParser {

  implicit class RichHSLFSlide(slide: HSLFSlide) {
    // Extract content from legacy PowerPoint slides (.ppt)
    def extractHSLFSlideContent: Seq[HTMLElement] = {
      val title = Option(slide.getTitle).getOrElse("")
      val titleElement = if (title.nonEmpty) {
        Seq(
          HTMLElement(elementType = ElementType.TITLE, content = title, metadata = mutable.Map()))
      } else Seq()

      val content: Seq[HTMLElement] = slide.getShapes.asScala.flatMap {
        case textShape: HSLFTextShape =>
          textShape.getTextParagraphs.asScala.flatMap { paragraph =>
            val isBullet = paragraph.isBullet
            val bulletSymbol = Option(paragraph.getBulletChar).getOrElse("")
            val paragraphText = paragraph.getTextRuns.asScala.map(_.getRawText).mkString("")

            if (isBullet) {
              Some(
                HTMLElement(
                  elementType = ElementType.LIST_ITEM,
                  content = s"$bulletSymbol $paragraphText",
                  metadata = mutable.Map()))
            } else if (paragraphText.nonEmpty) {
              Some(
                HTMLElement(
                  elementType = ElementType.NARRATIVE_TEXT,
                  content = paragraphText,
                  metadata = mutable.Map()))
            } else {
              None
            }
          }

        case table: HSLFTable =>
          val cellElements = (0 until table.getNumberOfRows).flatMap { rowIndex =>
            (0 until table.getNumberOfColumns).map { colIndex =>
              val cellContent =
                Option(table.getCell(rowIndex, colIndex)).map(_.getText).getOrElse("").trim
              HTMLElement(
                elementType = ElementType.TABLE,
                content = cellContent,
                metadata =
                  mutable.Map("tableLocation" -> s"(${rowIndex.toString}, ${colIndex.toString})"))
            }
          }

          cellElements

        case _ => Seq()
      }

      titleElement ++ content
    }
  }

  implicit class RichXSLFSlide(slide: XSLFSlide) {

    def extractXSLFSlideContent(inferTableStructure: Boolean): Seq[HTMLElement] = {
      val title = Option(slide.getTitle).getOrElse("")
      val titleElement = if (title.nonEmpty) {
        Seq(
          HTMLElement(elementType = ElementType.TITLE, content = title, metadata = mutable.Map()))
      } else Seq()

      val content: Seq[HTMLElement] = slide.getShapes.asScala.flatMap {
        case textShape: XSLFTextShape
            if textShape.getText != null &&
              textShape.getText != title =>
          textShape.getTextParagraphs.asScala.map { paragraph =>
            val isBullet = paragraph.isBullet
            val bulletSymbol = Option(paragraph.getBulletCharacter).getOrElse("")
            val paragraphText = paragraph.getText
            if (isBullet) {
              HTMLElement(
                elementType = ElementType.LIST_ITEM,
                content = s"$bulletSymbol $paragraphText",
                metadata = mutable.Map())
            } else {
              HTMLElement(
                elementType = ElementType.NARRATIVE_TEXT,
                content = paragraphText,
                metadata = mutable.Map())
            }
          }
        case table: XSLFTable =>
          val cellElements = table.getRows.asScala.zipWithIndex.flatMap { case (row, rowIndex) =>
            row.getCells.asScala.zipWithIndex.map { case (cell, colIndex) =>
              val cellContent = Option(cell.getText).getOrElse("").trim // Extract cell content
              HTMLElement(
                elementType = ElementType.TABLE,
                content = cellContent,
                metadata =
                  mutable.Map("tableLocation" -> s"(${rowIndex.toString}, ${colIndex.toString})"))
            }
          }
          if (inferTableStructure) {
            val tableHtml = buildTableHtml(table)
            val htmlElement = HTMLElement("HTML", tableHtml, mutable.Map("element" -> "table"))
            cellElements ++ Seq(htmlElement)
          } else {
            cellElements
          }
        case _ => Seq()
      }

      titleElement ++ content
    }

  }

  private def buildTableHtml(table: XSLFTable): String = {
    val rowsHtml = table.getRows.asScala
      .map { row =>
        val cellsHtml = row.getCells.asScala
          .map { cell =>
            val cellText = Option(cell.getText).getOrElse("").trim
            s"<td>$cellText</td>"
          }
          .mkString("")
        s"<tr>$cellsHtml</tr>"
      }
      .mkString("")
    s"<table>$rowsHtml</table>"
  }

}
