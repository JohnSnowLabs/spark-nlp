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

import com.johnsnowlabs.nlp.IAnnotation
import com.johnsnowlabs.reader.util.HasPdfToTextProperties
import com.johnsnowlabs.reader.util.pdf._
import com.johnsnowlabs.reader.util.pdf.schema.{MappingMatrix, PageMatrix}
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.text.PDFTextStripper
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, lit, posexplode_outer, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

import java.io.{ByteArrayOutputStream, PrintWriter, StringWriter}
import scala.util.{Failure, Success, Try}

/** Extract text from PDF document to a single string or to several strings per each page. Input
  * is a column with binary representation of PDF document. For the output it generates column
  * with text and page number. Explode each page as separate row if split to page enabled.
  *
  * It can be configured with the following properties:
  *   - pageNumCol: Page number output column name.
  *   - originCol: Input column name with original path of file.
  *   - partitionNum: Number of partitions. By default, it is set to 0.
  *   - storeSplittedPdf: Force to store bytes content of split pdf. By default, it is set to
  *     `false`.
  *   - splitPage: Enable/disable splitting per page to identify page numbers and improve
  *     performance. By default, it is set to `true`.
  *   - onlyPageNum: Extract only page numbers. By default, it is set to `false`.
  *   - textStripper: Text stripper type used for output layout and formatting.
  *   - sort: Enable/disable sorting content on the page. By default, it is set to `false`.
  *
  * ==Example==
  * {{{
  *     val pdfToText = new PdfToText()
  *       .setStoreSplittedPdf(true)
  *       .setSplitPage(true)
  *     val filesDf = spark.read.format("binaryFile").load("Documents/files/pdf")
  *     val pipelineModel = new Pipeline()
  *       .setStages(Array(pdfToText))
  *       .fit(filesDf)
  *
  *     val pdfDf = pipelineModel.transform(filesDf)
  *
  * pdfDf.show()
  * +--------------------+--------------------+------+--------------------+
  * |                path|    modificationTime|length|                text|
  * +--------------------+--------------------+------+--------------------+
  * |file:/Users/paula...|2025-05-15 11:33:...| 25803|This is a Title \...|
  * |file:/Users/paula...|2025-05-15 11:33:...| 15629|                  \n|
  * |file:/Users/paula...|2025-05-15 11:33:...| 15629|                  \n|
  * |file:/Users/paula...|2025-05-15 11:33:...| 15629|                  \n|
  * |file:/Users/paula...|2025-05-15 11:33:...|  9487|   This is a page.\n|
  * |file:/Users/paula...|2025-05-15 11:33:...|  9487|This is another p...|
  * |file:/Users/paula...|2025-05-15 11:33:...|  9487| Yet another page.\n|
  * |file:/Users/paula...|2025-05-15 11:56:...|  1563|Hello, this is li...|
  * +--------------------+--------------------+------+--------------------+
  *
  * pdfDf.printSchema()
  * root
  *  |-- path: string (nullable = true)
  *  |-- modificationTime: timestamp (nullable = true)
  *  |-- length: long (nullable = true)
  *  |-- text: string (nullable = true)
  *  |-- height_dimension: integer (nullable = true)
  *  |-- width_dimension: integer (nullable = true)
  *  |-- content: binary (nullable = true)
  *  |-- exception: string (nullable = true)
  *  |-- pagenum: integer (nullable = true)
  * }}}
  */
class PdfToText(override val uid: String)
    extends Transformer
    with DefaultParamsWritable
    with HasInputValidator
    with HasInputCol
    with HasOutputCol
    with HasLocalProcess
    with PdfToTextTrait
    with HasPdfToTextProperties {

  def this() = this(Identifiable.randomUID("PDF_TO_TEXT_TRANSFORMER"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  protected def outputDataType: StructType = new StructType()
    .add($(outputCol), StringType)
    .add("positions", PageMatrix.dataType)
    .add("height_dimension", IntegerType)
    .add("width_dimension", IntegerType)
    .add($(inputCol), BinaryType)
    .add("exception", StringType)
    .add($(pageNumCol), IntegerType)

  override def transformSchema(schema: StructType): StructType = {
    // Add the return fields
    validateInputCol(schema, $(inputCol), BinaryType)
    validateInputCol(schema, $(originCol), StringType)
    schema
      .add(StructField($(outputCol), StringType, nullable = false))
      .add(StructField($(pageNumCol), IntegerType, nullable = false))
  }

  /** @param value Name of input annotation col */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @param value Name of extraction output col */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  setDefault(inputCol -> "content", outputCol -> "text")

  private def transformUDF: UserDefinedFunction = udf(
    (path: String, content: Array[Byte], exception: String) => {
      doProcess(content, exception)
    },
    ArrayType(outputDataType))

  private def doProcess(
      content: Array[Byte],
      exception: String): Seq[(String, Seq[PageMatrix], Int, Int, Array[Byte], String, Int)] = {
    val pagesTry = Try(
      pdfToText(
        content,
        $(onlyPageNum),
        $(splitPage),
        $(storeSplittedPdf),
        $(sort),
        $(textStripper),
        $(extractCoordinates),
        $(normalizeLigatures)))

    pagesTry match {
      case Failure(e) =>
        log.error("Pdf load error during text extraction")
        val sw = new StringWriter
        e.printStackTrace(new PrintWriter(sw))
        log.error(sw.toString)
        log.error(pagesTry.toString)
        val errMessage = e.toString + " " + e.getMessage
        Seq(("", Seq(), -1, -1, Array(), exception.concatException(s"PdfToText: $errMessage"), 0))
      case Success(content) =>
        content
    }
  }

  override def transform(df: Dataset[_]): DataFrame = {
    transformSchema(df.schema)

    val selCols1 = df.columns
      .filterNot(_ == $(inputCol))
      .map(col) :+ posexplode_outer(
      transformUDF(
        df.col($(originCol)),
        df.col($(inputCol)),
        if (df.columns.contains("exception")) {
          col("exception")
        } else {
          lit(null)
        }))
      .as(Seq("tmp_num", "tmp_result"))
    val selCols = df.columns
      .filterNot(_ == $(inputCol))
      .map(col) :+ col("tmp_result.*")

    var result = df.select(selCols1: _*)
    result = result
      .select(selCols: _*)
    $(partitionNum) match {
      case 0 => result
      case _ => result.repartition($(partitionNum))
    }
  }

  override def localProcess(
      input: Array[Map[String, Seq[IAnnotation]]]): Array[Map[String, Seq[IAnnotation]]] = {
    input.flatMap { case lightRecord =>
      val pdfs = lightRecord.getOrElse(
        getOrDefault(inputCol),
        throw new RuntimeException(s"Column not found ${getOrDefault(inputCol)}"))

      pdfs flatMap {
        case BinaryFile(bytes, path) =>
          doProcess(bytes, path).zipWithIndex.map {
            case ((text, pageMatrix, _, _, content, exception, _), pageNum) =>
              val metadata =
                Map("exception" -> exception, "sourcePath" -> path, "pageNum" -> pageNum.toString)

              val result = lightRecord ++ Map(
                getOutputCol -> Seq(OcrText(text, metadata, content)),
                getOrDefault(pageNumCol) -> Seq(PageNum(pageNum)))

              if ($(extractCoordinates))
                result ++ Map("positions" -> pageMatrix.map(pm => PositionsOutput(pm.mapping)))
              else
                result

            case _ => lightRecord.chainExceptions(s"Wrong Input in $uid")
          }
        case _ => Seq(lightRecord.chainExceptions(s"Wrong Input in $uid"))
      }
    }
  }

}

trait PdfToTextTrait extends Logging with PdfUtils {

  /*
   * extracts a text layer from a PDF.
   */
  private def extractText(
      document: => PDDocument,
      startPage: Int,
      endPage: Int,
      sort: Boolean,
      textStripper: String): Seq[String] = {
    val pdfTextStripper: PDFTextStripper = textStripper match {
      case TextStripperType.PDF_LAYOUT_TEXT_STRIPPER =>
        val stripper = new PDFLayoutTextStripper()
        stripper.setIsSort(sort)
        stripper
      case _ => new PDFTextStripper
    }
    pdfTextStripper.setStartPage(startPage + 1)
    pdfTextStripper.setEndPage(endPage + 1)
    Seq(pdfTextStripper.getText(document))
  }

  def pdfToText(
      content: Array[Byte],
      onlyPageNum: Boolean,
      splitPage: Boolean,
      storeSplittedPdf: Boolean,
      sort: Boolean,
      textStripper: String,
      extractCoordinates: Boolean,
      normalizeLigatures: Boolean = false)
      : Seq[(String, Seq[PageMatrix], Int, Int, Array[Byte], String, Int)] = {
    val validPdf = checkAndFixPdf(content)
    val pdfDoc = PDDocument.load(validPdf)
    val numPages = pdfDoc.getNumberOfPages
    log.info(s"Number of pages $numPages")
    require(numPages >= 1, "pdf input stream cannot be empty")
    val result = if (!onlyPageNum) {
      pdfboxMethod(
        pdfDoc,
        0,
        numPages - 1,
        content,
        splitPage,
        storeSplittedPdf,
        sort,
        textStripper,
        extractCoordinates,
        normalizeLigatures = normalizeLigatures)
    } else {
      Range(1, numPages + 1).map(pageNum => ("", null, 1, 1, null, null, pageNum))
    }
    pdfDoc.close()
    log.info("Close pdf")
    result
  }

  private def pdfboxMethod(
      pdfDoc: => PDDocument,
      startPage: Int,
      endPage: Int,
      content: Array[Byte],
      splitPage: Boolean,
      storeSplittedPdf: Boolean,
      sort: Boolean,
      textStripper: String,
      extractCoordinates: Boolean,
      normalizeCoordinates: Boolean = true,
      normalizeLigatures: Boolean = false)
      : Seq[(String, Seq[PageMatrix], Int, Int, Array[Byte], String, Int)] = {
    lazy val out: ByteArrayOutputStream = new ByteArrayOutputStream()
    if (splitPage) {
      Range(startPage, endPage + 1).flatMap(pagenum =>
        extractText(pdfDoc, pagenum, pagenum, sort, textStripper)
          .map { text =>
            out.reset()
            val outputDocument = new PDDocument()
            val page = pdfDoc.getPage(pagenum)
            val splittedPdf = if (storeSplittedPdf) {
              outputDocument.importPage(page)
              outputDocument.save(out)
              outputDocument.close()
              out.toByteArray
            } else null
            val coordinates =
              if (extractCoordinates)
                getCoordinates(pdfDoc, pagenum, pagenum, normalizeCoordinates, normalizeLigatures)
              else null
            (
              text,
              coordinates,
              page.getMediaBox.getHeight.toInt,
              page.getMediaBox.getWidth.toInt,
              splittedPdf,
              null,
              pagenum)
          })
    } else {
      val text = extractText(pdfDoc, startPage, endPage, sort, textStripper).mkString(
        System.lineSeparator())
      val heightDimension = pdfDoc.getPage(startPage).getMediaBox.getHeight.toInt
      val widthDimension = pdfDoc.getPage(startPage).getMediaBox.getWidth.toInt
      val coordinates =
        if (extractCoordinates)
          getCoordinates(pdfDoc, startPage, endPage, normalizeCoordinates, normalizeLigatures)
        else null
      Seq(
        (
          text,
          coordinates,
          heightDimension,
          widthDimension,
          if (storeSplittedPdf) content else null,
          null,
          0))
    }
  }

  private def getCoordinates(
      doc: => PDDocument,
      startPage: Int,
      endPage: Int,
      normalizeOutput: Boolean = true,
      normalizeLigatures: Boolean = true): Seq[PageMatrix] = {
    import scala.collection.JavaConverters._
    val unicodeUtils = new UnicodeUtils
    Range(startPage, endPage + 1).map(pagenum => {
      val (_, pHeight) = getPageDims(pagenum, doc)
      val stripper = new CustomStripper
      stripper.setStartPage(pagenum + 1)
      stripper.setEndPage(pagenum + 1)
      stripper.getText(doc)
      val line = stripper.lines.asScala.flatMap(_.textPositions.asScala)

      val mappings = line.toArray.map(p => {
        MappingMatrix(
          p.toString,
          p.getTextMatrix.getTranslateX,
          if (normalizeOutput) pHeight - p.getTextMatrix.getTranslateY - p.getHeightDir
          else p.getTextMatrix.getTranslateY,
          p.getWidth,
          p.getHeightDir,
          0,
          "pdf")
      })

      val coordinates =
        if (normalizeLigatures) unicodeUtils.normalizeLigatures(mappings) else mappings
      PageMatrix(coordinates)
    })
  }

  private def getPageDims(numPage: Int, document: PDDocument) = {
    val page = document.getPage(numPage).getMediaBox
    (page.getWidth, page.getHeight)
  }
}

object PdfToText extends DefaultParamsReadable[PdfToText] {
  override def load(path: String): PdfToText = super.load(path)
}
