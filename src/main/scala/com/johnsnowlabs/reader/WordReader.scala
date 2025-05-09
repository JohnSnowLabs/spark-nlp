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
import com.johnsnowlabs.partition.util.PartitionHelper.datasetWithBinaryFile
import com.johnsnowlabs.reader.util.DocParser.RichParagraph
import com.johnsnowlabs.reader.util.DocxParser.{
  RichXWPFDocument,
  RichXWPFParagraph,
  RichXWPFTable
}
import org.apache.poi.hwpf.HWPFDocument
import org.apache.poi.xwpf.usermodel.{XWPFDocument, XWPFParagraph, XWPFTable}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import java.io.{ByteArrayInputStream, IOException}
import scala.collection.JavaConverters._
import scala.collection.mutable

/** Class to read and parse word files.
  *
  * @param storeContent
  *   Whether to include the raw file content in the output DataFrame as a separate 'content'
  *   column, alongside the structured output. By default, it is set to false.
  * @param includePageBreaks
  *   Whether to detect and tag content with page break metadata. In Word documents, this includes
  *   manual and section breaks. In Excel files, this includes page breaks based on column
  *   boundaries. By default, it is set to false.
  * @param inferTableStructure
  *   Whether to generate an HTML table representation from structured table content. When
  *   enabled, a full <table> element is added alongside cell-level elements, based on row and
  *   column layout. By default, it is set to false.
  *
  * ==Example==
  * {{{
  * val docDirectory = "home/user/word-directory"
  * val wordReader = new WordReader()
  * val wordDf = wordReader.doc(docDirectory)
  * }}}
  *
  * {{{
  * wordDf.select("doc").show(false)
  * +----------------------------------------------------------------------------------------------------------------------------------------------------+
  * |doc                                                                                                                                                 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
  * +----------------------------------------------------------------------------------------------------------------------------------------------------+
  * |[{Table, Header Col 1, {}}, {Table, Header Col 2, {}}, {Table, Lorem ipsum, {}}, {Table, A Link example, {}}, {NarrativeText, Dolor sit amet, {}}]  |
  * +----------------------------------------------------------------------------------------------------------------------------------------------------+
  *
  * wordDf.printSchema()
  * root
  *  |-- path: string (nullable = true)
  *  |-- doc: array (nullable = true)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- elementType: string (nullable = true)
  *  |    |    |-- content: string (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  *
  * }}}
  */

class WordReader(
    storeContent: Boolean = false,
    includePageBreaks: Boolean = false,
    inferTableStructure: Boolean = false)
    extends Serializable {

  private lazy val spark = ResourceHelper.spark

  private var outputColumn = "doc"

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Output column name cannot be empty.")
    outputColumn = value
    this
  }

  def getOutputColumn: String = outputColumn

  def doc(filePath: String): DataFrame = {
    if (ResourceHelper.validFile(filePath)) {
      val wordDf = datasetWithBinaryFile(spark, filePath)
        .withColumn(outputColumn, parseWordUDF(col("content")))
      if (storeContent) wordDf.select("path", outputColumn, "content")
      else wordDf.select("path", outputColumn)
    } else throw new IllegalArgumentException(s"Invalid filePath: $filePath")
  }

  private val parseWordUDF = udf((data: Array[Byte]) => {
    parseDoc(data)
  })

  def docToHTMLElement(content: Array[Byte]): Seq[HTMLElement] = {
    parseDoc(content)
  }

  // Constants for file type identification
  private val ZipMagicNumberFirstByte: Byte =
    0x50.toByte // First byte of ZIP files, indicating .docx
  private val ZipMagicNumberSecondByte: Byte =
    0x4b.toByte // Second byte of ZIP files, indicating .docx
  private val OleMagicNumber: Array[Byte] =
    Array(0xd0.toByte, 0xcf.toByte, 0x11.toByte, 0xe0.toByte) // Bytes indicating .doc

  private var pageBreak = 0

  private def isDocxFile(content: Array[Byte]): Boolean = {
    content.length > 1 && content(0) == ZipMagicNumberFirstByte && content(
      1) == ZipMagicNumberSecondByte
  }

  private def isDocFile(content: Array[Byte]): Boolean = {
    content.length >= 4 && content.slice(0, 4).sameElements(OleMagicNumber)
  }

  private def parseDoc(content: Array[Byte]): Seq[HTMLElement] = {
    pageBreak = 0
    val docInputStream = new ByteArrayInputStream(content)
    try {
      if (isDocxFile(content)) {
        val document = new XWPFDocument(docInputStream)
        val headers = document.extractHeaders.map { header =>
          HTMLElement(ElementType.HEADER, header, mutable.Map())
        }
        val footers = document.extractFooters.map { footer =>
          HTMLElement(ElementType.FOOTER, footer, mutable.Map())
        }
        val docElements = parseDocxToElements(document)
        headers ++ docElements ++ footers
      } else if (isDocFile(content)) {
        val document = new HWPFDocument(docInputStream)
        val docElements = parseDocToElements(document)
        docElements
      } else {
        Seq(HTMLElement(ElementType.UNCATEGORIZED_TEXT, "Unknown file format", mutable.Map()))
      }
    } catch {
      case e: IOException =>
        throw new IOException(s"Error e: ${e.getMessage}")
    } finally {
      docInputStream.close()
    }
  }

  private def parseDocxToElements(document: XWPFDocument): Seq[HTMLElement] = {

    val elements = document.getBodyElements.asScala.flatMap {
      case paragraph: XWPFParagraph =>
        processParagraph(paragraph, "paragraph")

      case table: XWPFTable =>
        processTable(table)

      case _ => None
    }

    elements
  }

  private def processParagraph(
      paragraph: XWPFParagraph,
      source: String,
      tableLocation: mutable.Map[String, String] = mutable.Map()): Option[HTMLElement] = {
    val text = paragraph.getText.trim
    if (text.isEmpty) None
    else {
      val metadata = mutable.Map[String, String]()

      if (includePageBreaks) {
        val isBreak = paragraph.isCustomPageBreak || paragraph.isSectionBreak
        if (isBreak) {
          pageBreak += 1
          metadata += ("pageBreak" -> pageBreak.toString)
        }
      }

      if (tableLocation.nonEmpty) {
        metadata ++= tableLocation
      }

      val elementType = paragraph match {
        case p if p.isTitle => ElementType.TITLE
        case p if p.isListItem => ElementType.LIST_ITEM
        case _ => if (source == "table") ElementType.TABLE else ElementType.NARRATIVE_TEXT
      }
      Some(HTMLElement(elementType, text, metadata))
    }
  }

  private def processTable(table: XWPFTable): Seq[HTMLElement] = {
    val tableHtml = if (inferTableStructure) Some(table.processAsHtml) else None

    val tableElements: Seq[HTMLElement] = table.getRows.asScala.zipWithIndex.flatMap {
      case (row, rowIndex) =>
        row.getTableCells.asScala.zipWithIndex.flatMap { case (cell, cellIndex) =>
          val tableLocation = mutable.Map("tableLocation" -> s"($rowIndex, $cellIndex)")
          cell.getParagraphs.asScala.flatMap { paragraph =>
            processParagraph(paragraph, "table", tableLocation)
          }
        }
    }

    if (tableHtml.isDefined) {
      val htmlElement =
        HTMLElement(ElementType.HTML, tableHtml.get, mutable.Map.empty[String, String])
      tableElements :+ htmlElement
    } else tableElements

  }

  private def parseDocToElements(document: HWPFDocument): Seq[HTMLElement] = {

    val paragraphs = document.getRange
    val elements = (0 until paragraphs.numParagraphs).flatMap { i =>
      val paragraph = paragraphs.getParagraph(i)
      val text = paragraph.text.trim
      if (text.isEmpty) None
      else {
        val metadata = mutable.Map[String, String]()
        paragraph match {
          case p if p.isInTable(paragraphs) =>
            val tableText = p.tableText(paragraphs).getOrElse("")
            Some(HTMLElement(ElementType.TABLE, tableText, metadata))
          case p if p.isTitle => Some(HTMLElement(ElementType.TITLE, text, metadata))
          case p if p.isListItem => Some(HTMLElement(ElementType.LIST_ITEM, text, metadata))
          case _ => Some(HTMLElement(ElementType.NARRATIVE_TEXT, text, metadata))
        }

      }
    }

    elements
  }

}
