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

import org.apache.poi.hwpf.usermodel.{Paragraph, Range, Table}

import scala.util.Try

object DocParser {

  implicit class RichParagraph(paragraph: Paragraph) {

    def isTitle: Boolean = {

      val text = paragraph.text.trim
      val isUppercase = (text == text.toUpperCase) && !isListItem
      (containsBoldText && containsCapitalizedWords) || (isUppercase && isCenterAligned)
    }

    private def containsBoldText: Boolean = {
      val characterRuns = (0 until paragraph.numCharacterRuns).map(paragraph.getCharacterRun)
      characterRuns.exists(_.isBold)
    }

    private def isCenterAligned: Boolean = {
      paragraph.getJustification == 1
    }

    private def containsCapitalizedWords: Boolean = {
      val words = paragraph.text.trim.split("\\s+")
      words.forall(word => word.nonEmpty && word.head.isUpper)
    }

    def isListItem: Boolean = {
      val text = paragraph.text.trim
      text.startsWith("•") || text.startsWith("–") || text.startsWith("*") ||
      paragraph.getIndentFromLeft > 0
    }

    def isInTable(range: Range): Boolean = {
      Try(range.getTable(paragraph)).isSuccess
    }

    def tableText(range: Range): Option[String] = {
      Try {
        val table = range.getTable(paragraph)
        val rows = (0 until table.numRows).map(table.getRow)
        val cellTexts = rows.flatMap(row =>
          (0 until row.numCells)
            .flatMap { cellIndex =>
              val cell = row.getCell(cellIndex)
              (0 until cell.numParagraphs).map(cell.getParagraph)
            }
            .map(_.text.trim))
        cellTexts.mkString(" | ") // Join cell texts with a separator
      }.toOption
    }
  }

}
