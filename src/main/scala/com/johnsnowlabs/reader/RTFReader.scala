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
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import java.awt.AWTError
import java.io.{ByteArrayInputStream, StringReader}
import java.util.UUID
import javax.swing.text.{DefaultStyledDocument, Element, StyleConstants}
import javax.swing.text.rtf.RTFEditorKit
import scala.collection.mutable

/** Class to read and parse Rich Text Format (RTF) files.
  *
  * The reader extracts paragraph-level content from `.rtf` documents and maps it into
  * `HTMLElement`s, classifying paragraphs as titles, list items, or narrative text using the same
  * structural conventions used by the other readers.
  *
  * @param storeContent
  *   Whether to include the raw RTF bytes in the output DataFrame as a separate `content` column.
  *   Default is `false`.
  * @param titleLengthSize
  *   Maximum character length used when deciding whether a paragraph should be classified as a
  *   title. Default is `50`.
  */
class RTFReader(storeContent: Boolean = false, titleLengthSize: Int = 50) extends Serializable {

  private lazy val spark = ResourceHelper.spark

  private var outputColumn = "rtf"

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Output column name cannot be empty.")
    outputColumn = value
    this
  }

  def getOutputColumn: String = outputColumn

  def rtf(filePath: String): DataFrame = {
    if (ResourceHelper.validFile(filePath)) {
      val rtfDf = datasetWithBinaryFile(spark, filePath)
        .withColumn(outputColumn, parseRtfUDF(col("content")))
      if (storeContent) rtfDf.select("path", outputColumn, "content")
      else rtfDf.select("path", outputColumn)
    } else {
      throw new IllegalArgumentException(s"Invalid filePath: $filePath")
    }
  }

  def rtfToHTMLElement(content: String): Seq[HTMLElement] = {
    parseSafely {
      parseRtfDocument { (kit, document) =>
        val reader = new StringReader(content)
        try kit.read(reader, document, 0)
        finally reader.close()
      }
    }
  }

  def rtfToHTMLElement(content: Array[Byte]): Seq[HTMLElement] = {
    parseSafely {
      parseRtfDocument { (kit, document) =>
        val inputStream = new ByteArrayInputStream(content)
        try kit.read(inputStream, document, 0)
        finally inputStream.close()
      }
    }
  }

  private val parseRtfUDF = udf((content: Array[Byte]) => rtfToHTMLElement(content))

  private case class ParagraphStyle(
      hasBold: Boolean,
      hasItalic: Boolean,
      hasUnderline: Boolean,
      maxFontSize: Int)

  private def parseSafely(parseFn: => Seq[HTMLElement]): Seq[HTMLElement] = {
    try {
      parseFn
    } catch {
      case e: Exception =>
        Seq(
          HTMLElement(ElementType.ERROR, s"Could not parse RTF: ${e.getMessage}", mutable.Map()))
    }
  }

  private def parseRtfDocument(
      readerFn: (RTFEditorKit, DefaultStyledDocument) => Unit): Seq[HTMLElement] = {
    val document = loadStyledDocument(readerFn)
    extractParagraphElements(document)
  }

  private def loadStyledDocument(
      readerFn: (RTFEditorKit, DefaultStyledDocument) => Unit): DefaultStyledDocument = {

    def buildDocument(): DefaultStyledDocument = {
      val kit = new RTFEditorKit()
      val document = new DefaultStyledDocument()
      readerFn(kit, document)
      document
    }

    try {
      buildDocument()
    } catch {
      case e: AWTError if needsHeadlessFallback(e) =>
        enableHeadlessAwt()
        buildDocument()
    }
  }

  private def needsHeadlessFallback(error: AWTError): Boolean = {
    Option(error.getMessage)
      .exists(message =>
        message.contains("X11 window server") || message.contains("Assistive Technology"))
  }

  private def enableHeadlessAwt(): Unit = RTFReader.synchronized {
    if (System.getProperty("java.awt.headless") == null) {
      System.setProperty("java.awt.headless", "true")
    }
    if (System.getProperty("javax.accessibility.assistive_technologies") == null) {
      System.setProperty("javax.accessibility.assistive_technologies", "")
    }
  }

  private def extractParagraphElements(document: DefaultStyledDocument): Seq[HTMLElement] = {
    val root = document.getDefaultRootElement
    val elements = mutable.ArrayBuffer[HTMLElement]()
    var sentenceIndex = 0
    var currentParentId: Option[String] = None

    (0 until root.getElementCount).foreach { paragraphIndex =>
      val paragraph = root.getElement(paragraphIndex)
      val text = extractParagraphText(document, paragraph)

      if (text.nonEmpty) {
        val style = summarizeParagraphStyle(paragraph)
        val elementId = UUID.randomUUID().toString
        val metadata = mutable.Map(
          "paragraph" -> paragraphIndex.toString,
          "sentence" -> sentenceIndex.toString,
          "element_id" -> elementId)

        if (style.hasBold) metadata("bold") = "true"
        if (style.hasItalic) metadata("italic") = "true"
        if (style.hasUnderline) metadata("underline") = "true"
        if (style.maxFontSize > 0) metadata("fontSize") = style.maxFontSize.toString

        val elementType = classifyParagraph(text, style)
        elementType match {
          case ElementType.TITLE =>
            currentParentId = Some(elementId)
          case _ =>
            currentParentId.foreach(parentId => metadata("parent_id") = parentId)
        }

        elements += HTMLElement(elementType, text, metadata)
        sentenceIndex += 1
      }
    }

    elements
  }

  private def extractParagraphText(
      document: DefaultStyledDocument,
      paragraph: Element): String = {
    val text =
      document.getText(
        paragraph.getStartOffset,
        paragraph.getEndOffset - paragraph.getStartOffset)
    text.replace("\r", "").replace("\n", "").trim
  }

  private def summarizeParagraphStyle(paragraph: Element): ParagraphStyle = {
    val runAttributes = (0 until paragraph.getElementCount)
      .map(paragraph.getElement)
      .map(_.getAttributes)

    ParagraphStyle(
      hasBold = runAttributes.exists(StyleConstants.isBold),
      hasItalic = runAttributes.exists(StyleConstants.isItalic),
      hasUnderline = runAttributes.exists(StyleConstants.isUnderline),
      maxFontSize = runAttributes.map(StyleConstants.getFontSize).foldLeft(0)(Math.max))
  }

  private def classifyParagraph(text: String, style: ParagraphStyle): String = {
    if (isListItem(text)) {
      ElementType.LIST_ITEM
    } else if (isTitle(text, style)) {
      ElementType.TITLE
    } else {
      ElementType.NARRATIVE_TEXT
    }
  }

  private def isTitle(text: String, style: ParagraphStyle): Boolean = {
    val trimmed = text.trim
    trimmed.nonEmpty && !isListItem(trimmed) && (isTitleCandidate(trimmed) ||
      (style.hasBold && style.maxFontSize >= 12 && isCapitalized(trimmed)))
  }

  private def isTitleCandidate(text: String): Boolean = {
    val trimmed = text.trim
    if (trimmed.isEmpty) return false

    val isAllUpper = trimmed.forall(c => !c.isLetter || c.isUpper)
    val isShort = trimmed.length <= titleLengthSize
    val hasLetters = trimmed.exists(_.isLetter)
    (isAllUpper || isCapitalized(trimmed)) && isShort && hasLetters
  }

  private def isCapitalized(text: String): Boolean = {
    text
      .split("\\s+")
      .filter(_.nonEmpty)
      .forall(word => word.dropWhile(ch => !ch.isLetter).headOption.exists(_.isUpper))
  }

  private def isListItem(text: String): Boolean = {
    text.matches("""^(?:[\u2022\u25E6\u25AA\u25CF\u2013\u2014*\-]|\d+[.)]|[a-zA-Z][.)])\s+.+$""")
  }
}
