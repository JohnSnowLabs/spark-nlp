/*
 * Copyright 2017-2025 John Snow Labs
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
import com.johnsnowlabs.reader.util.ImageParser
import org.apache.pdfbox.contentstream.PDFGraphicsStreamEngine
import org.apache.pdfbox.cos.COSName
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.graphics.image.PDImage
import org.apache.pdfbox.text.{PDFTextStripper, TextPosition}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import java.awt.geom.Point2D
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.util.UUID
import javax.imageio.ImageIO
import scala.collection.JavaConverters._
import scala.collection.mutable

/** Class to parse and read PDF files.
  *
  * @param titleThreshold
  *   Minimum font size threshold used as part of heuristic rules to detect title elements based
  *   on formatting (e.g., bold, centered, capitalized). By default, it is set to 18.
  * @param storeContent
  *   Whether to include the raw file content in the output DataFrame as a separate 'content'
  *
  * pdfPath: this is a path to a directory of HTML files or a path to an HTML file E.g.
  * "path/pdf/files"
  *
  * ==Example==
  * {{{
  * val path = "./pdf-files/pdf-doc.pdf"
  * val PdfReader = new PdfReader()
  * val pdfDF = PdfReader.read(url)
  * }}}
  *
  * {{{
  * pdfDF.show()
  * +--------------------+--------------------+
  * |                path|                html|
  * +--------------------+--------------------+
  * |file:/content/htm...|[{Title, My First...|
  * +--------------------+--------------------+
  *
  * pdfDF.printSchema()
  * root
  *  |-- path: string (nullable = true)
  *  |-- pdf: array (nullable = true)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- elementType: string (nullable = true)
  *  |    |    |-- content: string (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  * }}}
  * For more examples please refer to this
  * [[https://github.com/JohnSnowLabs/spark-nlp/examples/python/reader/SparkNLP_PDF_Reader_Demo.ipynb notebook]].
  */
class PdfReader(
    storeContent: Boolean = false,
    titleThreshold: Double = 18.0,
    readAsImage: Boolean = false)
    extends Serializable {

  private val paragraphSpacingY = 25
  private lazy val spark = ResourceHelper.spark
  private var outputColumn = "pdf"

  private case class EmbeddedPdfImage(
      pageIndex: Int,
      pageNumber: Int,
      coordX: Int,
      coordY: Int,
      width: Int,
      height: Int,
      bytes: Array[Byte],
      format: String)

  private case class DrawnImagePosition(
      x: Float,
      y: Float,
      width: Float,
      height: Float,
      image: BufferedImage)

  def setOutputColumn(name: String): this.type = {
    require(name.nonEmpty, "Output column name cannot be empty.")
    outputColumn = name
    this
  }
  def getOutputColumn: String = outputColumn

  def pdf(filePath: String): DataFrame = {
    if (!ResourceHelper.validFile(filePath))
      throw new IllegalArgumentException(s"Invalid filePath: $filePath")

    val binaryDF = datasetWithBinaryFile(spark, filePath)
    val withElements = binaryDF.withColumn(outputColumn, parsePdfUDF(col("content")))
    if (storeContent) withElements.select("path", outputColumn, "content")
    else withElements.select("path", outputColumn)
  }

  private val parsePdfUDF = udf((data: Array[Byte]) => pdfToHTMLElement(data))

  def pdfToHTMLElement(content: Array[Byte]): Seq[HTMLElement] = {
    try {
      if (readAsImage) {
        transformPdfToImages(content)
      } else {
        val pdfDoc = PDDocument.load(content)
        try {
          extractElementsFromPdf(pdfDoc)
        } finally {
          pdfDoc.close()
        }
      }
    } catch {
      case e: Exception =>
        Seq(
          HTMLElement(ElementType.ERROR, s"Could not parse PDF: ${e.getMessage}", mutable.Map()))
    }
  }

  private def extractElementsFromPdf(pdfDoc: PDDocument): Seq[HTMLElement] = {
    val collectedElements = mutable.ListBuffer[HTMLElement]()
    var currentParentId: Option[String] = None
    var paragraphIndex = 0

    val textStripper = new PDFTextStripper() {
      override def writeString(
          text: String,
          textPositions: java.util.List[TextPosition]): Unit = {
        val lineGroups = groupTextPositionsByLine(textPositions)
        val lineElements = lineGroups.flatMap { case (_, linePositions) =>
          val element =
            classifyLineElement(linePositions, getCurrentPageNo, currentParentId, paragraphIndex)
          if (element.nonEmpty) paragraphIndex += 1
          element
        }

        // Update parentId when encountering titles
        lineElements.foreach { elem =>
          collectedElements += elem
          if (elem.elementType == ElementType.TITLE)
            currentParentId = Some(elem.metadata("element_id"))
        }

      }
    }
    textStripper.setSortByPosition(true)
    textStripper.setStartPage(1)
    textStripper.setEndPage(pdfDoc.getNumberOfPages)
    textStripper.getText(pdfDoc)
    collectedElements
  }

  private def groupTextPositionsByLine(
      textPositions: java.util.List[TextPosition]): Seq[(Int, Seq[TextPosition])] = {
    val yTolerance = 2f // Potential parameter, since needs to experiment to fit your PDFs
    textPositions.asScala.groupBy(tp => (tp.getY / yTolerance).round).toSeq.sortBy(_._1)
  }

  private def classifyLineElement(
      linePositions: Seq[TextPosition],
      pageNumber: Int,
      currentParentId: Option[String],
      paragraphIndex: Int): Option[HTMLElement] = {
    val lineText = linePositions.map(_.getUnicode).mkString.trim
    if (lineText.isEmpty) return None

    val averageFontSize = linePositions.map(_.getFontSize).sum / linePositions.size
    val mostCommonFontName = linePositions.groupBy(_.getFont.getName).maxBy(_._2.size)._1
    val isTitleLine = isTitle(averageFontSize, mostCommonFontName)
    val elementType =
      if (isTitleLine) ElementType.TITLE else ElementType.NARRATIVE_TEXT

    val pageY = linePositions.map(_.getY).min.round
    val metadata =
      mutable.Map(
        "pageNumber" -> pageNumber.toString,
        "element_id" -> UUID.randomUUID().toString,
        "paragraph_index" -> paragraphIndex.toString,
        "paragraph_y" -> (paragraphIndex * paragraphSpacingY).toString,
        "page_y" -> pageY.toString)
    // Assign parent_id only for narrative text or non-titles
    if (!isTitleLine) currentParentId.foreach(pid => metadata("parent_id") = pid)
    Some(HTMLElement(elementType, lineText, metadata))
  }

  private def isTitle(fontSize: Double, fontName: String): Boolean = {
    fontSize >= titleThreshold || fontName.toLowerCase.contains("bold")
  }

  private def transformPdfToImages(content: Array[Byte]): Seq[HTMLElement] = {
    val pageImages = ImageParser.renderPdfFile(content) // Map[Int, Option[BufferedImage]]
    val embeddedImagesByPage = extractEmbeddedImages(content).groupBy(_.pageIndex)

    pageImages.toSeq
      .sortBy(_._1)
      .flatMap { case (pageIndex, renderedPageOpt) =>
        val embeddedImages = embeddedImagesByPage.getOrElse(pageIndex, Seq.empty)
        if (embeddedImages.nonEmpty) {
          embeddedImages.sortBy(img => (img.coordY, img.coordX)).map(buildImageElement)
        } else {
          renderedPageOpt.toSeq.map { bufferedImage =>
            val pageNumber = pageIndex + 1
            val encoded = encodeBufferedImage(bufferedImage)
            val coordY = pageIndex * bufferedImage.getHeight
            val metadata = mutable.Map(
              "pageNumber" -> pageNumber.toString,
              "coord" -> s"{x:0,y:$coordY}",
              "format" -> encoded._2,
              "image_type" -> "floating",
              "width" -> bufferedImage.getWidth.toString,
              "height" -> bufferedImage.getHeight.toString,
              "element_id" -> UUID.randomUUID().toString)

            HTMLElement(
              elementType = ElementType.IMAGE,
              content = "",
              metadata = metadata,
              binaryContent = Some(encoded._1))
          }
        }
      }
      .toSeq
  }

  private def buildImageElement(image: EmbeddedPdfImage): HTMLElement = {
    val metadata = mutable.Map(
      "pageNumber" -> image.pageNumber.toString,
      "coord" -> s"{x:${image.coordX},y:${image.coordY}}",
      "format" -> image.format,
      "image_type" -> "floating",
      "width" -> image.width.toString,
      "height" -> image.height.toString,
      "element_id" -> UUID.randomUUID().toString)

    HTMLElement(
      elementType = ElementType.IMAGE,
      content = "",
      metadata = metadata,
      binaryContent = Some(image.bytes))
  }

  private def extractEmbeddedImages(content: Array[Byte]): Seq[EmbeddedPdfImage] = {
    val pdfDoc = PDDocument.load(content)
    try {
      (0 until pdfDoc.getNumberOfPages).flatMap { pageIndex =>
        val page = pdfDoc.getPage(pageIndex)
        val pageNumber = pageIndex + 1
        val pageHeight = page.getMediaBox.getHeight
        val collector = new PdfPageImagePositionCollector(page)
        collector.processPage(page)

        collector.positions.flatMap { position =>
          val topY = pageHeight - (position.y + position.height)
          val coordX = math.max(0, math.round(position.x))
          val coordY = math.max(0, math.round(topY))
          val width = math.max(1, math.round(position.width))
          val height = math.max(1, math.round(position.height))
          val encoded = encodeBufferedImage(position.image)

          if (encoded._1.nonEmpty)
            Some(
              EmbeddedPdfImage(
                pageIndex = pageIndex,
                pageNumber = pageNumber,
                coordX = coordX,
                coordY = coordY,
                width = width,
                height = height,
                bytes = encoded._1,
                format = encoded._2))
          else None
        }
      }
    } catch {
      case _: Exception => Seq.empty
    } finally {
      pdfDoc.close()
    }
  }

  private def encodeBufferedImage(image: BufferedImage): (Array[Byte], String) = {
    if (image == null) return (Array.emptyByteArray, "jpg")

    val jpgBytes = imageToBytes(image, "jpg")
    if (jpgBytes.nonEmpty) (jpgBytes, "jpg")
    else {
      val pngBytes = imageToBytes(image, "png")
      if (pngBytes.nonEmpty) (pngBytes, "png")
      else (Array.emptyByteArray, "jpg")
    }
  }

  private def imageToBytes(image: BufferedImage, format: String): Array[Byte] = {
    val baos = new ByteArrayOutputStream()
    try {
      val written = ImageIO.write(image, format, baos)
      if (written) baos.toByteArray else Array.emptyByteArray
    } catch {
      case _: Exception => Array.emptyByteArray
    } finally {
      baos.close()
    }
  }

  private class PdfPageImagePositionCollector(page: PDPage)
      extends PDFGraphicsStreamEngine(page) {

    private val currentPoint = new Point2D.Float()
    private val collectedPositions = mutable.ListBuffer[DrawnImagePosition]()

    def positions: Seq[DrawnImagePosition] = collectedPositions.toSeq

    override def drawImage(pdImage: PDImage): Unit = {
      val bufferedImage = pdImage.getImage
      if (bufferedImage != null) {
        val transformMatrix = getGraphicsState.getCurrentTransformationMatrix
        val width = math.abs(transformMatrix.getScalingFactorX)
        val height = math.abs(transformMatrix.getScalingFactorY)
        if (width > 0 && height > 0) {
          collectedPositions += DrawnImagePosition(
            x = transformMatrix.getTranslateX,
            y = transformMatrix.getTranslateY,
            width = width,
            height = height,
            image = bufferedImage)
        }
      }
    }

    override def appendRectangle(p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D): Unit = {}
    override def clip(windingRule: Int): Unit = {}
    override def moveTo(x: Float, y: Float): Unit = currentPoint.setLocation(x, y)
    override def lineTo(x: Float, y: Float): Unit = currentPoint.setLocation(x, y)
    override def curveTo(x1: Float, y1: Float, x2: Float, y2: Float, x3: Float, y3: Float): Unit =
      currentPoint.setLocation(x3, y3)
    override def getCurrentPoint: Point2D = currentPoint
    override def closePath(): Unit = {}
    override def endPath(): Unit = {}
    override def strokePath(): Unit = {}
    override def fillPath(windingRule: Int): Unit = {}
    override def fillAndStrokePath(windingRule: Int): Unit = {}
    override def shadingFill(shadingName: COSName): Unit = {}
  }

}
