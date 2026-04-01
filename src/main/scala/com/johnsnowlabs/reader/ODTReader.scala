/*
 * Copyright 2017-2026 John Snow Labs
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
import com.johnsnowlabs.reader.util.HTMLParser
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import java.nio.charset.StandardCharsets
import java.util.UUID
import java.util.zip.ZipInputStream
import scala.collection.mutable
import scala.util.Try
import scala.xml.{Elem, Node, Text, Utility, XML}

/** Class to read and parse ODT files.
  *
  * Public behavior matches [[WordReader]] so it can be used in the same ingestion flows.
  */
class ODTReader(
    storeContent: Boolean = false,
    includePageBreaks: Boolean = false,
    inferTableStructure: Boolean = false,
    outputFormat: String = "json-table")
    extends Serializable {

  private lazy val spark = ResourceHelper.spark

  private var outputColumn = "doc"
  private var pageBreak = 0
  private var pendingPageBreak: Option[Int] = None
  private var currentParentId: Option[String] = None
  private var paragraphIndexCounter = 0

  private val paragraphSpacingY = 25

  private case class OdtStyle(
      name: String,
      parentName: Option[String],
      isHeading: Boolean,
      breakBeforePage: Boolean,
      breakAfterPage: Boolean,
      pageNumber: Option[String],
      masterPageName: Option[String])

  private case class OdtState(var tableCounter: Int = 0, var imageCounter: Int = 0)

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Output column name cannot be empty.")
    outputColumn = value
    this
  }

  def getOutputColumn: String = outputColumn

  def doc(filePath: String): DataFrame = {
    if (ResourceHelper.validFile(filePath)) {
      val odtDf =
        datasetWithBinaryFile(spark, filePath).withColumn(
          outputColumn,
          parseOdtUDF(col("content")))
      if (storeContent) odtDf.select("path", outputColumn, "content")
      else odtDf.select("path", outputColumn)
    } else throw new IllegalArgumentException(s"Invalid filePath: $filePath")
  }

  def docToHTMLElement(content: Array[Byte]): Seq[HTMLElement] = parseOdt(content)

  private val parseOdtUDF = udf((content: Array[Byte]) => parseOdt(content))

  private def parseOdt(content: Array[Byte]): Seq[HTMLElement] = {
    pageBreak = 0
    pendingPageBreak = None
    currentParentId = None
    paragraphIndexCounter = 0

    try {
      val zipEntries = readZipEntries(content)
      if (!isOdtFile(zipEntries)) {
        return Seq(
          HTMLElement(ElementType.UNCATEGORIZED_TEXT, "Unknown file format", mutable.Map()))
      }

      val contentXml = readXmlEntry(zipEntries, "content.xml")
      val stylesXmlOpt = zipEntries.get("styles.xml").map(readXmlBytes)
      val headers = extractHeaderFooter(stylesXmlOpt, "header")
        .map(text => HTMLElement(ElementType.HEADER, text, mutable.Map()))
      val footers = extractHeaderFooter(stylesXmlOpt, "footer")
        .map(text => HTMLElement(ElementType.FOOTER, text, mutable.Map()))

      val styleIndex = buildStyleIndex(contentXml +: stylesXmlOpt.toSeq)
      val officeText = findOfficeText(contentXml)
        .getOrElse(throw new IllegalArgumentException("ODT content.xml is missing office:text"))
      val state = OdtState()
      val images = mutable.ArrayBuffer[HTMLElement]()
      val previousCoords = mutable.Map[(Int, Int), Int]()
      val docElements =
        processNodes(
          childElements(officeText),
          state,
          styleIndex,
          zipEntries,
          images,
          previousCoords)

      headers ++ docElements ++ footers ++ images.toSeq
    } catch {
      case e: Exception =>
        Seq(
          HTMLElement(ElementType.ERROR, s"Could not parse ODT: ${e.getMessage}", mutable.Map()))
    }
  }

  private def processNodes(
      nodes: Seq[Node],
      state: OdtState,
      styleIndex: Map[String, OdtStyle],
      zipEntries: Map[String, Array[Byte]],
      images: mutable.ArrayBuffer[HTMLElement],
      previousCoords: mutable.Map[(Int, Int), Int],
      insideList: Boolean = false): Seq[HTMLElement] = {
    nodes.flatMap {
      case elem: Elem =>
        elem.label match {
          case "h" =>
            val paragraphIndex = nextParagraphIndex()
            processTextElement(
              elem,
              paragraphIndex,
              state,
              styleIndex,
              zipEntries,
              images,
              previousCoords,
              forceTitle = true)
          case "p" =>
            val paragraphIndex = nextParagraphIndex()
            processTextElement(
              elem,
              paragraphIndex,
              state,
              styleIndex,
              zipEntries,
              images,
              previousCoords,
              isListItem = insideList)
          case "list" =>
            processList(elem, state, styleIndex, zipEntries, images, previousCoords)
          case "table" =>
            val paragraphIndex = nextParagraphIndex()
            processTable(
              elem,
              paragraphIndex,
              state,
              styleIndex,
              zipEntries,
              images,
              previousCoords)
          case "soft-page-break" =>
            if (includePageBreaks) registerStandalonePageBreak()
            Seq.empty
          case _ =>
            processNodes(
              childElements(elem),
              state,
              styleIndex,
              zipEntries,
              images,
              previousCoords,
              insideList)
        }
      case _ => Seq.empty
    }
  }

  private def processList(
      listNode: Elem,
      state: OdtState,
      styleIndex: Map[String, OdtStyle],
      zipEntries: Map[String, Array[Byte]],
      images: mutable.ArrayBuffer[HTMLElement],
      previousCoords: mutable.Map[(Int, Int), Int]): Seq[HTMLElement] = {
    childElements(listNode).flatMap {
      case elem: Elem if elem.label == "list-item" || elem.label == "list-header" =>
        processNodes(
          childElements(elem),
          state,
          styleIndex,
          zipEntries,
          images,
          previousCoords,
          insideList = true)
      case elem: Elem =>
        processNodes(
          childElements(elem),
          state,
          styleIndex,
          zipEntries,
          images,
          previousCoords,
          insideList = true)
      case _ => Seq.empty
    }
  }

  private def processTextElement(
      element: Elem,
      paragraphIndex: Int,
      state: OdtState,
      styleIndex: Map[String, OdtStyle],
      zipEntries: Map[String, Array[Byte]],
      images: mutable.ArrayBuffer[HTMLElement],
      previousCoords: mutable.Map[(Int, Int), Int],
      isListItem: Boolean = false,
      forceTitle: Boolean = false,
      tableLocation: mutable.Map[String, String] = mutable.Map()): Option[HTMLElement] = {
    collectImagesFromNode(element, paragraphIndex, state, zipEntries, images, previousCoords)

    val metadata = mutable.Map[String, String]()
    applyPageBreakMetadata(element, metadata, styleIndex)

    val text = extractText(element).trim
    if (text.isEmpty) {
      return None
    }

    val elementId = newUUID()
    metadata("element_id") = elementId
    metadata("paragraph_index") = paragraphIndex.toString
    metadata("paragraph_y") = (paragraphIndex * paragraphSpacingY).toString
    if (tableLocation.nonEmpty) metadata ++= tableLocation

    val isTitle = forceTitle || isHeadingStyle(element, styleIndex)
    val elementType =
      if (isTitle) {
        currentParentId = Some(elementId)
        ElementType.TITLE
      } else if (isListItem) {
        currentParentId.foreach(pid => metadata("parent_id") = pid)
        ElementType.LIST_ITEM
      } else {
        currentParentId.foreach(pid => metadata("parent_id") = pid)
        if (tableLocation.nonEmpty) ElementType.TABLE else ElementType.NARRATIVE_TEXT
      }

    Some(HTMLElement(elementType, text, metadata))
  }

  private def processTable(
      tableNode: Elem,
      paragraphIndex: Int,
      state: OdtState,
      styleIndex: Map[String, OdtStyle],
      zipEntries: Map[String, Array[Byte]],
      images: mutable.ArrayBuffer[HTMLElement],
      previousCoords: mutable.Map[(Int, Int), Int]): Seq[HTMLElement] = {
    state.tableCounter += 1
    val tableId = newUUID()
    val rows = expandRows(tableNode)
    val tableMetadata = mutable.Map[String, String](
      "domPath" -> s"/table[${state.tableCounter}]",
      "orderTableIndex" -> state.tableCounter.toString,
      "element_id" -> tableId)

    val structuredTable = if (inferTableStructure) Some(buildHtmlTable(rows)) else None
    val tableElement = structuredTable.map { html =>
      outputFormat match {
        case "html-table" =>
          HTMLElement(ElementType.HTML, html, tableMetadata)
        case "json-table" =>
          val tableElem = HTMLParser.parseFirstTableElement(html)
          val jsonString = HTMLParser.tableElementToJson(tableElem)
          HTMLElement(ElementType.JSON, jsonString, tableMetadata)
        case _ =>
          HTMLElement(ElementType.TABLE, rows.flatten.mkString(" ").trim, tableMetadata)
      }
    }

    val tableCells = childElements(tableNode).zipWithIndex.flatMap {
      case (row: Elem, rowIndex) if row.label == "table-row" =>
        expandCells(row).zipWithIndex.flatMap { case (cell, cellIndex) =>
          val tableLocation = mutable.Map(
            "element_id" -> newUUID(),
            "parent_id" -> tableId,
            "domPath" -> s"/table[${state.tableCounter}]/row[${rowIndex + 1}]/cell[${cellIndex + 1}]",
            "orderTableIndex" -> state.tableCounter.toString)
          processTableCellContent(
            cell,
            paragraphIndex,
            state,
            styleIndex,
            zipEntries,
            images,
            previousCoords,
            tableLocation)
        }
      case _ => Seq.empty
    }

    tableCells ++ tableElement.toSeq
  }

  private def processTableCellContent(
      cell: Elem,
      paragraphIndex: Int,
      state: OdtState,
      styleIndex: Map[String, OdtStyle],
      zipEntries: Map[String, Array[Byte]],
      images: mutable.ArrayBuffer[HTMLElement],
      previousCoords: mutable.Map[(Int, Int), Int],
      tableLocation: mutable.Map[String, String]): Seq[HTMLElement] = {
    childElements(cell).flatMap {
      case elem: Elem if elem.label == "p" =>
        processTextElement(
          elem,
          paragraphIndex,
          state,
          styleIndex,
          zipEntries,
          images,
          previousCoords,
          tableLocation = tableLocation)
      case elem: Elem if elem.label == "h" =>
        processTextElement(
          elem,
          paragraphIndex,
          state,
          styleIndex,
          zipEntries,
          images,
          previousCoords,
          forceTitle = true,
          tableLocation = tableLocation)
      case elem: Elem if elem.label == "list" =>
        processList(elem, state, styleIndex, zipEntries, images, previousCoords).map {
          case html if html.metadata.contains("parent_id") => html
          case html =>
            html.copy(metadata = html.metadata ++ tableLocation)
        }
      case elem: Elem =>
        processTableCellContent(
          elem,
          paragraphIndex,
          state,
          styleIndex,
          zipEntries,
          images,
          previousCoords,
          tableLocation)
      case _ => Seq.empty
    }
  }

  private def collectImagesFromNode(
      node: Node,
      paragraphIndex: Int,
      state: OdtState,
      zipEntries: Map[String, Array[Byte]],
      images: mutable.ArrayBuffer[HTMLElement],
      previousCoords: mutable.Map[(Int, Int), Int]): Unit = {
    node.descendant.collect { case elem: Elem if elem.label == "frame" => elem }.foreach {
      frame =>
        childElements(frame).find(_.label == "image").foreach { imageNode =>
          state.imageCounter += 1
          val href = attr(imageNode, "href").getOrElse("")
          val normalizedHref = normalizeZipPath(href)
          val imageBytes = zipEntries.get(normalizedHref)
          val imageType =
            attr(frame, "anchor-type")
              .filter(_.nonEmpty)
              .map {
                case "as-char" => "inline"
                case _ => "floating"
              }
              .getOrElse("floating")
          val (x, y) =
            deduplicateCoords(extractFrameCoords(frame, paragraphIndex), previousCoords)
          val fileName =
            normalizedHref.split("/").lastOption.getOrElse(s"image-${state.imageCounter}")
          val metadata = mutable.Map(
            "domPath" -> s"/img[${state.imageCounter}]",
            "orderImageIndex" -> state.imageCounter.toString,
            "format" -> fileName.split("\\.").lastOption.getOrElse("bin"),
            "coord" -> s"{x:$x,y:$y}",
            "image_type" -> imageType)

          if (imageType == "inline") {
            metadata("element_id") = newUUID()
          }

          images += HTMLElement(ElementType.IMAGE, fileName, metadata, imageBytes)
        }
    }
  }

  private def applyPageBreakMetadata(
      element: Elem,
      metadata: mutable.Map[String, String],
      styleIndex: Map[String, OdtStyle]): Unit = {
    if (!includePageBreaks) {
      return
    }

    pendingPageBreak.foreach { breakNumber =>
      metadata("pageBreak") = breakNumber.toString
      pendingPageBreak = None
    }

    val styleBreak =
      resolveParagraphStyle(element, styleIndex).exists { style =>
        style.breakBeforePage || style.breakAfterPage || style.pageNumber.exists(
          _ != "1") || style.masterPageName.exists(_.nonEmpty)
      }
    val inlineBreak = element.descendant.exists {
      case elem: Elem => elem.label == "soft-page-break"
      case _ => false
    }

    if (styleBreak || inlineBreak) {
      pageBreak += 1
      metadata("pageBreak") = pageBreak.toString
      currentParentId = None
    }
  }

  private def registerStandalonePageBreak(): Unit = {
    pageBreak += 1
    pendingPageBreak = Some(pageBreak)
    currentParentId = None
  }

  private def resolveParagraphStyle(
      element: Elem,
      styleIndex: Map[String, OdtStyle]): Option[OdtStyle] = {
    attr(element, "style-name").flatMap { styleName =>
      def resolve(name: String, seen: Set[String]): Option[OdtStyle] = {
        if (seen.contains(name)) None
        else {
          styleIndex.get(name).map { style =>
            style.parentName
              .flatMap(parent => resolve(parent, seen + name))
              .map(parent => mergeStyles(parent, style))
              .getOrElse(style)
          }
        }
      }
      resolve(styleName, Set.empty)
    }
  }

  private def mergeStyles(parent: OdtStyle, child: OdtStyle): OdtStyle = {
    child.copy(
      parentName = child.parentName.orElse(parent.parentName),
      isHeading = child.isHeading || parent.isHeading,
      breakBeforePage = child.breakBeforePage || parent.breakBeforePage,
      breakAfterPage = child.breakAfterPage || parent.breakAfterPage,
      pageNumber = child.pageNumber.orElse(parent.pageNumber),
      masterPageName = child.masterPageName.orElse(parent.masterPageName))
  }

  private def isHeadingStyle(element: Elem, styleIndex: Map[String, OdtStyle]): Boolean = {
    if (element.label == "h") {
      true
    } else {
      resolveParagraphStyle(element, styleIndex).exists(_.isHeading)
    }
  }

  private def buildStyleIndex(xmlRoots: Seq[Elem]): Map[String, OdtStyle] = {
    val styles = xmlRoots.flatMap(_.descendant.collect {
      case elem: Elem if elem.prefix == "style" && elem.label == "style" =>
        val name = attr(elem, "name")
        val family = attr(elem, "family")
        val parentName = attr(elem, "parent-style-name")
        val titleHints =
          Seq(name, attr(elem, "display-name"), parentName).flatten.exists { value =>
            val lower = value.toLowerCase
            lower.contains("heading") || lower.contains("title")
          }
        val paragraphProperties = childElements(elem).find { child =>
          child.prefix == "style" && child.label == "paragraph-properties"
        }
        val breakBeforePage =
          paragraphProperties.flatMap(prop => attr(prop, "break-before")).contains("page")
        val breakAfterPage =
          paragraphProperties.flatMap(prop => attr(prop, "break-after")).contains("page")
        val pageNumber = paragraphProperties.flatMap(prop => attr(prop, "page-number"))
        val masterPageName = attr(elem, "master-page-name")
        if (family.contains("paragraph") && name.nonEmpty)
          Some(
            OdtStyle(
              name.get,
              parentName,
              isHeading = titleHints,
              breakBeforePage = breakBeforePage,
              breakAfterPage = breakAfterPage,
              pageNumber = pageNumber,
              masterPageName = masterPageName))
        else None
    })
    styles.flatten.map(style => style.name -> style).toMap
  }

  private def expandRows(tableNode: Elem): Seq[Seq[String]] = {
    childElements(tableNode).filter(_.label == "table-row").flatMap { row =>
      val rowRepeat = attr(row, "number-rows-repeated").flatMap(safeToInt).getOrElse(1)
      val cells = expandCells(row).map(extractCellText)
      Seq.fill(rowRepeat)(cells)
    }
  }

  private def expandCells(row: Elem): Seq[Elem] = {
    childElements(row).flatMap {
      case cell: Elem if cell.label == "table-cell" || cell.label == "covered-table-cell" =>
        val repeat = attr(cell, "number-columns-repeated").flatMap(safeToInt).getOrElse(1)
        Seq.fill(repeat)(cell)
      case _ => Seq.empty
    }
  }

  private def extractCellText(cell: Elem): String = {
    childElements(cell)
      .flatMap {
        case elem: Elem if elem.label == "p" || elem.label == "h" =>
          Some(extractText(elem).trim).filter(_.nonEmpty)
        case elem: Elem if elem.label == "list" =>
          childElements(elem)
            .flatMap(item => childElements(item))
            .collect {
              case paragraph: Elem if paragraph.label == "p" => extractText(paragraph).trim
            }
            .filter(_.nonEmpty)
        case _ => Seq.empty
      }
      .mkString(" ")
  }

  private def buildHtmlTable(rows: Seq[Seq[String]]): String = {
    val body = rows.map { row =>
      val cells = row.map(cell => s"<td>${Utility.escape(cell)}</td>").mkString
      s"<tr>$cells</tr>"
    }.mkString
    s"<table>$body</table>"
  }

  private def extractHeaderFooter(stylesXmlOpt: Option[Elem], label: String): Seq[String] = {
    stylesXmlOpt.toSeq
      .flatMap(_.descendant.collect {
        case elem: Elem if elem.prefix == "style" && elem.label == label =>
          extractText(elem).trim
      })
      .filter(_.nonEmpty)
      .distinct
  }

  private def extractText(node: Node): String = node match {
    case text: Text => text.text
    case elem: Elem if elem.label == "line-break" => "\n"
    case elem: Elem if elem.label == "tab" => "\t"
    case elem: Elem if elem.label == "s" =>
      " " * attr(elem, "c").flatMap(safeToInt).getOrElse(1)
    case elem: Elem if elem.label == "soft-page-break" => ""
    case elem: Elem => elem.child.map(extractText).mkString
    case _ => ""
  }

  private def extractFrameCoords(frame: Elem, paragraphIndex: Int): (Int, Int) = {
    val x = attr(frame, "x").map(parseLengthToPx).getOrElse {
      attr(frame, "anchor-type") match {
        case Some("as-char") => 0
        case _ => 50
      }
    }
    val y = attr(frame, "y").map(parseLengthToPx).getOrElse(paragraphIndex * paragraphSpacingY)
    (x, y)
  }

  private def deduplicateCoords(
      coords: (Int, Int),
      previousCoords: mutable.Map[(Int, Int), Int]): (Int, Int) = {
    val duplicateCount = previousCoords.getOrElse(coords, 0)
    val adjusted = (coords._1, coords._2 + (duplicateCount * 20))
    previousCoords.update(coords, duplicateCount + 1)
    adjusted
  }

  private def parseLengthToPx(value: String): Int = {
    val lengthPattern = """([0-9.]+)\s*(cm|mm|in|pt|px)?""".r
    value.trim match {
      case lengthPattern(number, unit) =>
        val numeric = number.toDouble
        unit match {
          case "cm" => Math.round(numeric * 37.8).toInt
          case "mm" => Math.round(numeric * 3.78).toInt
          case "in" => Math.round(numeric * 96.0).toInt
          case "pt" => Math.round(numeric * 1.333).toInt
          case _ => Math.round(numeric).toInt
        }
      case _ => 0
    }
  }

  private def isOdtFile(zipEntries: Map[String, Array[Byte]]): Boolean = {
    val mimeType = zipEntries
      .get("mimetype")
      .map(bytes => new String(bytes, StandardCharsets.UTF_8).trim)
    mimeType.contains("application/vnd.oasis.opendocument.text") || zipEntries.contains(
      "content.xml")
  }

  private def readZipEntries(content: Array[Byte]): Map[String, Array[Byte]] = {
    val zipEntries = mutable.Map[String, Array[Byte]]()
    val zipInputStream = new ZipInputStream(new ByteArrayInputStream(content))

    try {
      var entry = zipInputStream.getNextEntry
      while (entry != null) {
        if (!entry.isDirectory) {
          val buffer = new Array[Byte](4096)
          val output = new ByteArrayOutputStream()
          var bytesRead = zipInputStream.read(buffer)
          while (bytesRead != -1) {
            output.write(buffer, 0, bytesRead)
            bytesRead = zipInputStream.read(buffer)
          }
          zipEntries += normalizeZipPath(entry.getName) -> output.toByteArray
        }
        entry = zipInputStream.getNextEntry
      }
    } finally {
      zipInputStream.close()
    }

    zipEntries.toMap
  }

  private def readXmlEntry(zipEntries: Map[String, Array[Byte]], entryName: String): Elem = {
    val bytes =
      zipEntries.getOrElse(entryName, throw new IllegalArgumentException(s"Missing $entryName"))
    readXmlBytes(bytes)
  }

  private def readXmlBytes(bytes: Array[Byte]): Elem =
    XML.loadString(new String(bytes, StandardCharsets.UTF_8))

  private def findOfficeText(xml: Elem): Option[Elem] =
    xml.descendant.collectFirst {
      case elem: Elem if elem.prefix == "office" && elem.label == "text" => elem
    }

  private def childElements(node: Node): Seq[Elem] = node.child.collect { case elem: Elem =>
    elem
  }

  private def attr(node: Node, key: String): Option[String] = {
    node.attributes.asAttrMap
      .collectFirst {
        case (attrKey, value) if attrKey == key || attrKey.endsWith(s":$key") => value.trim
      }
      .filter(_.nonEmpty)
  }

  private def safeToInt(value: String): Option[Int] = Try(value.toInt).toOption

  private def normalizeZipPath(path: String): String =
    Option(path).getOrElse("").stripPrefix("./").stripPrefix("/")

  private def newUUID(): String = UUID.randomUUID().toString

  private def nextParagraphIndex(): Int = {
    val current = paragraphIndexCounter
    paragraphIndexCounter += 1
    current
  }
}
