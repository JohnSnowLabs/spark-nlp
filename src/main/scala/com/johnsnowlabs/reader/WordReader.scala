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
import com.johnsnowlabs.reader.util.HTMLParser
import org.apache.poi.hwpf.HWPFDocument
import org.apache.poi.xwpf.usermodel.{XWPFDocument, XWPFParagraph, XWPFTable}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import java.io.ByteArrayInputStream
import java.util.UUID
import scala.collection.JavaConverters._
import scala.collection.mutable

/** Class to read and parse Word files.
  *
  * @param storeContent
  *   Whether to include the raw file content in the output DataFrame as a separate `content`
  *   column, alongside the structured output. Default is `false`.
  * @param includePageBreaks
  *   Whether to detect and tag content with page break metadata. In Word documents, this includes
  *   manual and section breaks. In Excel files, this includes page breaks based on column
  *   boundaries. Default is `false`.
  * @param inferTableStructure
  *   Whether to generate an HTML table representation from structured table content. When
  *   enabled, a full table element is added alongside cell-level elements, based on row and
  *   column layout. Default is `false`.
  *
  * ==Example==
  * {{{
  * val docDirectory = "./word-files/fake_table.docx"
  * val wordReader = new WordReader()
  * val wordDf = wordReader.doc(docDirectory)
  *
  * wordDf.show()
  * +--------------------+--------------------+
  * |                path|                 doc|
  * +--------------------+--------------------+
  * |file:/content/wor...|[{Table, Header C...|
  * +--------------------+--------------------+
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
  * }}}
  * For more examples please refer to this
  * [[https://github.com/JohnSnowLabs/spark-nlp/examples/python/reader/SparkNLP_Word_Reader_Demo.ipynb notebook]].
  */
class WordReader(
    storeContent: Boolean = false,
    includePageBreaks: Boolean = false,
    inferTableStructure: Boolean = false,
    outputFormat: String = "json-table")
    extends Serializable {

  private lazy val spark = ResourceHelper.spark

  private var outputColumn = "doc"

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Output column name cannot be empty.")
    outputColumn = value
    this
  }

  def getOutputColumn: String = outputColumn

  /** @param filePath
    *   this is a path to a directory of word files or a path to a word file E.g.
    *   "path/word/files"
    *
    * @return
    *   Dataframe with parsed word doc content.
    */

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
  private var currentParentId: Option[String] = None
  private def newUUID(): String = UUID.randomUUID().toString

  private case class DocState(var tableCounter: Int = 0, var lastHeader: Option[String] = None)

  private def isDocxFile(content: Array[Byte]): Boolean = {
    content.length > 1 && content(0) == ZipMagicNumberFirstByte && content(
      1) == ZipMagicNumberSecondByte
  }

  private def isDocFile(content: Array[Byte]): Boolean = {
    content.length >= 4 && content.slice(0, 4).sameElements(OleMagicNumber)
  }

  private def parseDoc(content: Array[Byte]): Seq[HTMLElement] = {
    // Track element order and structure across the document
    val state = DocState()

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
        val docElements = parseDocxToElements(document, state)
        val images = extractImagesDocx(document, state)
        headers ++ docElements ++ footers ++ images
      } else if (isDocFile(content)) {
        val document = new HWPFDocument(docInputStream)
        val docElements = parseDocToElements(document, state)
        val images = extractImages(document, state)
        docElements ++ images
      } else {
        Seq(HTMLElement(ElementType.UNCATEGORIZED_TEXT, "Unknown file format", mutable.Map()))
      }
    } catch {
      case e: Exception =>
        Seq(
          HTMLElement(ElementType.ERROR, s"Could not parse Word: ${e.getMessage}", mutable.Map()))
    } finally {
      docInputStream.close()
    }
  }

  private def parseDocxToElements(document: XWPFDocument, state: DocState): Seq[HTMLElement] = {
    val elements = document.getBodyElements.asScala.flatMap {
      case paragraph: XWPFParagraph =>
        processParagraph(paragraph, "paragraph")

      case table: XWPFTable =>
        processTable(table, state)

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
      val elementId = newUUID()
      metadata("element_id") = elementId

      val style = Option(paragraph.getStyleID).getOrElse("").toLowerCase
      val isHeading = style.startsWith("heading") || style.startsWith("title")

      // Handle page breaks if needed
      if (includePageBreaks) {
        val isBreak = paragraph.isCustomPageBreak || paragraph.isSectionBreak
        if (isBreak) {
          pageBreak += 1
          metadata += ("pageBreak" -> pageBreak.toString)
          currentParentId = None
        }
      }

      if (tableLocation.nonEmpty) metadata ++= tableLocation

      val elementType = paragraph match {
        case _ if isHeading =>
          // Titles have no parent
          currentParentId = Some(elementId)
          ElementType.TITLE
        case p if p.isListItem =>
          currentParentId.foreach(pid => metadata("parent_id") = pid)
          ElementType.LIST_ITEM
        case _ =>
          currentParentId.foreach(pid => metadata("parent_id") = pid)
          if (source == "table") ElementType.TABLE else ElementType.NARRATIVE_TEXT
      }

      Some(HTMLElement(elementType, text, metadata))
    }
  }

  private def processTable(table: XWPFTable, state: DocState): Seq[HTMLElement] = {
    state.tableCounter += 1

    val tableHtml = if (inferTableStructure) Some(table.processAsHtml) else None
    val tableId = newUUID()

    val tableMetadata = mutable.Map[String, String](
      "domPath" -> s"/table[${state.tableCounter}]",
      "orderTableIndex" -> state.tableCounter.toString,
      "element_id" -> tableId)
    state.lastHeader.foreach(h => tableMetadata("nearestHeader") = h)

    val tableElement: Option[HTMLElement] = tableHtml.map { html =>
      outputFormat match {
        case "html-table" =>
          HTMLElement(ElementType.HTML, html, tableMetadata)
        case "json-table" =>
          val tableElem = HTMLParser.parseFirstTableElement(html)
          val jsonString = HTMLParser.tableElementToJson(tableElem)
          HTMLElement(ElementType.JSON, jsonString, tableMetadata)
        case _ =>
          HTMLElement(ElementType.TABLE, table.getText.trim, tableMetadata)
      }
    }

    val tableElements: Seq[HTMLElement] = table.getRows.asScala.zipWithIndex.flatMap {
      case (row, rowIndex) =>
        row.getTableCells.asScala.zipWithIndex.flatMap { case (cell, cellIndex) =>
          val tableLocation = mutable.Map(
            "element_id" -> newUUID(),
            "parent_id" -> tableId,
            "domPath" -> s"/table[${state.tableCounter}]/row[${rowIndex + 1}]/cell[${cellIndex + 1}]",
            "orderTableIndex" -> state.tableCounter.toString)
          state.lastHeader.foreach(h => tableLocation("nearestHeader") = h)
          cell.getParagraphs.asScala.flatMap { paragraph =>
            processParagraph(paragraph, "table", tableLocation)
          }
        }
    }

    tableElements ++ tableElement.toSeq
  }

  private def parseDocToElements(document: HWPFDocument, state: DocState): Seq[HTMLElement] = {
    val range = document.getRange
    var tableCounter = 0
    val elements = mutable.ArrayBuffer[HTMLElement]()

    for (i <- 0 until range.numParagraphs) {
      val paragraph = range.getParagraph(i)
      val text = paragraph.text.trim
      if (text.nonEmpty) {
        val metadata = mutable.Map[String, String](
          "element_id" -> newUUID()
        )

        if (paragraph.isInTable(range)) {
          tableCounter += 1
          val tableText = paragraph.tableText(range).getOrElse("")
          metadata("domPath") = s"/table[$tableCounter]"
          metadata("orderTableIndex") = tableCounter.toString
          state.lastHeader.foreach(h => metadata("nearestHeader") = h)
          elements += HTMLElement(ElementType.TABLE, tableText, metadata)
        } else {
          val elementType =
            if (paragraph.isTitle) ElementType.TITLE
            else if (paragraph.isListItem) ElementType.LIST_ITEM
            else ElementType.NARRATIVE_TEXT

          elements += HTMLElement(elementType, text, metadata)
        }
      }
    }

    elements
  }


  private def extractImagesDocx(document: XWPFDocument, state: DocState): Seq[HTMLElement] = {
    var imageIndex = 0

    document.getAllPictures.asScala.map { pic =>
      imageIndex += 1

      val metadata = mutable.Map[String, String](
        "domPath" -> s"/img[$imageIndex]",
        "orderImageIndex" -> imageIndex.toString,
        "element_id" -> newUUID(),
        "format" -> pic.suggestFileExtension,
        "imageType" -> pic.getPictureType.toString)
      state.lastHeader.foreach(h => metadata("nearestHeader") = h)

      val imageName = Option(pic.getFileName).getOrElse(s"image_$imageIndex")

      HTMLElement(
        elementType = ElementType.IMAGE,
        content = imageName,
        metadata = metadata,
        binaryContent = Some(pic.getData))
    }
  }

  private def extractImages(document: HWPFDocument, state: DocState): Seq[HTMLElement] = {
    var imageIndex = 0

    document.getPicturesTable.getAllPictures.asScala.map { pic =>
      imageIndex += 1
      val metadata = mutable.Map[String, String](
        "domPath" -> s"/img[$imageIndex]",
        "orderImageIndex" -> imageIndex.toString,
        "element_id" -> newUUID(),
        "format" -> pic.suggestFileExtension,
        "imageType" -> Option(pic.getMimeType).getOrElse("unknown")
      )

      state.lastHeader.foreach(h => metadata("nearestHeader") = h)

      val imageName = s"image_$imageIndex"

      HTMLElement(
        elementType = ElementType.IMAGE,
        content = imageName,
        metadata = metadata,
        binaryContent = Some(pic.getContent))
    }
  }


}
