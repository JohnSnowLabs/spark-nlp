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
import org.openxmlformats.schemas.drawingml.x2006.wordprocessingDrawing.{CTAnchor, CTInline}

import java.io.ByteArrayInputStream
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
  //TODO: Make paragraphSpacingY a parameter?
  private val paragraphSpacingY = 25 // heuristics px vertical increment per paragraph

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
    document.getBodyElements.asScala.zipWithIndex.flatMap {
      case (paragraph: XWPFParagraph, paragraphIndex) =>
        processParagraph(paragraph, "paragraph", paragraphIndex)

      case (table: XWPFTable, paragraphIndex) =>
        processTable(table, state, paragraphIndex)

      case _ => None
    }
  }

  private def processParagraph(
      paragraph: XWPFParagraph,
      source: String,
      paragraphIndex: Int,
      tableLocation: mutable.Map[String, String] = mutable.Map()): Option[HTMLElement] = {

    val text = paragraph.getText.trim
    if (text.isEmpty) None
    else {
      val metadata = mutable.Map[String, String]()
      val paragraphSpacingY = 25  // SAME heuristic used for images
      metadata("paragraph_index") = paragraphIndex.toString //TODO: Could be used as page number or to approximate it
      metadata("paragraph_y") = (paragraphIndex * paragraphSpacingY).toString

      val style = Option(paragraph.getStyleID).getOrElse("").toLowerCase
      val isHeading = style.startsWith("heading") || style.startsWith("title")

      // Handle page breaks if needed
      if (includePageBreaks) {
        val isBreak = paragraph.isCustomPageBreak || paragraph.isSectionBreak
        if (isBreak) {
          pageBreak += 1
          metadata += ("pageBreak" -> pageBreak.toString)
        }
      }

      if (tableLocation.nonEmpty) metadata ++= tableLocation

      val elementType = paragraph match {
        case _ if isHeading =>
          ElementType.TITLE
        case p if p.isListItem =>
          ElementType.LIST_ITEM
        case _ =>
          if (source == "table") ElementType.TABLE else ElementType.NARRATIVE_TEXT
      }

      Some(HTMLElement(elementType, text, metadata))
    }
  }

  private def processTable(table: XWPFTable, state: DocState, paragraphIndex: Int): Seq[HTMLElement] = {
    state.tableCounter += 1

    val tableHtml = if (inferTableStructure) Some(table.processAsHtml) else None

    val tableMetadata = mutable.Map[String, String](
      "domPath" -> s"/table[${state.tableCounter}]",
      "orderTableIndex" -> state.tableCounter.toString
    )
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
            "domPath" -> s"/table[${state.tableCounter}]/row[${rowIndex + 1}]/cell[${cellIndex + 1}]",
            "orderTableIndex" -> state.tableCounter.toString)
          state.lastHeader.foreach(h => tableLocation("nearestHeader") = h)
          cell.getParagraphs.asScala.flatMap { paragraph =>
            processParagraph(paragraph, "table", paragraphIndex, tableLocation)
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
        val metadata = mutable.Map[String, String]()

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

  /** Extracts images and computed coordinates (x, y, width, height) from a Word (.docx) document
    * using Apache POI.
    *
    * This function handles both <wp:anchor> (floating) and <wp:inline> (embedded) images. If
    * multiple anchors share identical (x, y) positions in the XML, a vertical offset (+20px) is
    * automatically applied to preserve spatial uniqueness for each image.
    *
    * @param document
    *   The XWPFDocument to process.
    * @param state
    *   Parsing state for headers, ordering, etc.
    * @return
    *   A sequence of HTMLElement objects with coordinate metadata.
    */
  private def extractImagesDocx(
                                 document: XWPFDocument,
                                 state: DocState
                               ): Seq[HTMLElement] = {

    val images = mutable.ArrayBuffer[HTMLElement]()
    var imageIndex = 0
    val previousCoords = mutable.Map[(Int, Int), Int]()

    document.getParagraphs.asScala.zipWithIndex.foreach {
      case (paragraph, paragraphIndex) =>
        paragraph.getRuns.asScala.zipWithIndex.foreach {
          case (run, runIndex) =>
            val ctrOpt = Option(run.getCTR)
            val drawings = ctrOpt.toSeq.flatMap(_.getDrawingList.asScala)

            drawings.foreach { drawing =>
              val anchors = drawing.getAnchorList.asScala
              val inlines = drawing.getInlineList.asScala

              // Handle floating images
              anchors.foreach { anchor =>
                run.getEmbeddedPictures.asScala.foreach { pic =>
                  imageIndex += 1

                  val coords = extractAnchorCoords(anchor, paragraphIndex)
                  val (x, y) = parseXY(coords("coord"), previousCoords)

                  val metadata = mutable.Map(
                    "domPath" -> s"/img[$imageIndex]",
                    "orderImageIndex" -> imageIndex.toString,
                    "format" -> pic.getPictureData.suggestFileExtension,
                    "coord" -> s"{x:$x,y:$y}",
                    "image_type" -> "floating"
                  )

                  images += HTMLElement(
                    ElementType.IMAGE,
                    pic.getPictureData.getFileName,
                    metadata,
                    Some(pic.getPictureData.getData)
                  )
                }
              }

              // Handle inline images
              inlines.foreach { inline =>
                run.getEmbeddedPictures.asScala.foreach { pic =>
                  imageIndex += 1

                  val coords = extractInlineCoords(inline, paragraphIndex, runIndex)
                  val (x, y) = parseXY(coords("coord"), previousCoords)

                  val metadata = mutable.Map(
                    "domPath" -> s"/img[$imageIndex]",
                    "orderImageIndex" -> imageIndex.toString,
                    "format" -> pic.getPictureData.suggestFileExtension,
                    "coord" -> s"{x:$x,y:$y}",
                    "image_type" -> "inline"
                  )

                  images += HTMLElement(
                    ElementType.IMAGE,
                    pic.getPictureData.getFileName,
                    metadata,
                    Some(pic.getPictureData.getData)
                  )
                }
              }
            }
        }
    }

    images
  }

  private def parseXY(
                       coordString: String,
                       previousCoords: mutable.Map[(Int, Int), Int]
                     ): (Int, Int) = {

    val coordPattern = """\{x:(\d+),y:(\d+)\}""".r

    val (x, y) = coordPattern
      .findFirstMatchIn(coordString)
      .map(m => (m.group(1).toInt, m.group(2).toInt))
      .getOrElse((0, 0))

    val duplicateCount = previousCoords.getOrElse((x, y), 0)
    val adjustedY = y + (duplicateCount * 20)

    previousCoords.update((x, y), duplicateCount + 1)

    (x, adjustedY)
  }



  /** Extracts approximate coordinates for floating (anchored) images in DOCX files.
    *
    * Word (OOXML) stores all positional data in **EMUs** (English Metric Units). EMU is a very
    * fine-grained unit — 1 inch = 914,400 EMUs, 1 cm = 360,000 EMUs.
    *
    * To convert EMUs to screen-like pixel coordinates, we use: 1 pixel = 9,525 EMUs (based on 96
    * DPI display standard)
    *
    * This function:
    *   - Converts EMU offsets (if available) into approximate pixels.
    *   - Falls back to alignment or paragraph-based heuristics if no explicit offsets are stored
    *     (common in "Move with text" or auto-anchored images).
    *
    * Returns coordinates as a compact JSON-like string: "{x:...,y:...}"
    *
    * @param anchor
    *   The CTAnchor element containing image positioning data.
    * @param paragraphIndex
    *   The paragraph index used for vertical approximation.
    * @return
    *   A map with one entry: "coord" -> "{x:...,y:...}".
    */
  private def extractAnchorCoords(anchor: CTAnchor, paragraphIndex: Int): Map[String, String] = {
    // Conversion constant: EMUs → pixels (1 inch = 914,400 EMUs; 1 px ≈ 9,525 EMUs)
    val EmusPerPixel = 9525
    val emuToPx = (v: Long) => (v / EmusPerPixel).toInt

    val posHOpt = Option(anchor.getPositionH)
    val posVOpt = Option(anchor.getPositionV)
    val simplePosOpt = Option(anchor.getSimplePos)

    // Extract numeric offsets (in EMUs)
    val xOffsetEmu = posHOpt.flatMap(ph => Option(ph.getPosOffset)).map(_.toLong).getOrElse(0L)
    val yOffsetEmu = posVOpt.flatMap(pv => Option(pv.getPosOffset)).map(_.toLong).getOrElse(0L)
    val xSimpleEmu = simplePosOpt.map(_.getX).getOrElse(0L)
    val ySimpleEmu = simplePosOpt.map(_.getY).getOrElse(0L)

    val hasPosOffset = xOffsetEmu != 0L || yOffsetEmu != 0L

    // Convert EMU → pixel (approx.)
    val xPx = emuToPx(if (hasPosOffset) xOffsetEmu else xSimpleEmu)
    val yPx = emuToPx(if (hasPosOffset) yOffsetEmu else ySimpleEmu)

    // Alignment and positioning hints
    val alignH =
      posHOpt.flatMap(ph => Option(ph.getAlign)).map(_.toString.toLowerCase).getOrElse("none")
    val relativeFromH =
      posHOpt.flatMap(ph => Option(ph.getRelativeFrom)).map(_.toString).getOrElse("unknown")

    // Fallback heuristics for missing offsets
    val alignmentCenterX = 300 // px for horizontally centered image
    val alignmentRightX = 600 // px for right-aligned image
    val pageRelativeX = 100 // px for "page" relative anchors
    val marginRelativeX = 50 // px for "margin" relative anchors
    val columnRelativeX = 25 // px for "column" relative anchors
    val paragraphRelativeX = 15 // px per paragraph index fallback

    // Decision hierarchy
    val finalX =
      if (hasPosOffset) xPx
      else if (alignH == "center") alignmentCenterX
      else if (alignH == "right") alignmentRightX
      else
        relativeFromH match {
          case "page" => pageRelativeX
          case "margin" => marginRelativeX
          case "column" => columnRelativeX
          case _ => paragraphIndex * paragraphRelativeX
        }

    val finalY =
      if (hasPosOffset) yPx
      else paragraphIndex * paragraphSpacingY

    val coordString = s"{x:$finalX,y:$finalY}"
    Map("coord" -> coordString, "image_type" -> "floating")
  }

  /** Approximates coordinates for inline images (CTInline) when Word does not explicitly store
    * positional data (common for embedded images).
    *
    * The approximation assumes:
    *   - ~20px vertical spacing per paragraph (line height)
    *   - ~8px horizontal spacing per run (average word spacing)
    *
    * These heuristics are used only for inline elements that flow with text. The result is
    * returned as a compact string in JSON-like form: "{x:approxX,y:approxY}"
    *
    * @param inline
    *   The CTInline image object.
    * @param paragraphIndex
    *   The paragraph index (used for Y approximation).
    * @param runIndex
    *   The run index (used for X approximation).
    * @return
    *   A map with a single entry: "coord" -> "{x:...,y:...}".
    */
  private def extractInlineCoords(
      inline: CTInline,
      paragraphIndex: Int,
      runIndex: Int): Map[String, String] = {
    val lineHeightPx = 20 // estimated vertical line height per paragraph
    val avgRunWidthPx = 8 // estimated horizontal width per run (word spacing)

    val approxY = paragraphIndex * lineHeightPx
    val approxX = runIndex * avgRunWidthPx

    val coordString = s"{x:$approxX,y:$approxY}"
    Map("coord" -> coordString, "image_type" -> "inline")
  }

  private def extractImages(document: HWPFDocument, state: DocState): Seq[HTMLElement] = {
    var imageIndex = 0

    document.getPicturesTable.getAllPictures.asScala.map { pic =>
      imageIndex += 1
      val metadata = mutable.Map[String, String](
        "domPath" -> s"/img[$imageIndex]",
        "orderImageIndex" -> imageIndex.toString,
        "format" -> pic.suggestFileExtension)

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
