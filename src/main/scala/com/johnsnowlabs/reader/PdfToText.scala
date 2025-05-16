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
import com.johnsnowlabs.reader.util.HasPdfProperties
import com.johnsnowlabs.reader.util.pdf._
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.text.PDFTextStripper
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, posexplode_outer, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

import java.io.ByteArrayOutputStream
import scala.util.{Failure, Success, Try}

class PdfToText(override val uid: String)
    extends Transformer
    with DefaultParamsWritable
    with HasInputValidator
    with HasInputCol
    with HasOutputCol
    with HasLocalProcess
    with PdfToTextTrait
    with HasPdfProperties {

  def this() = this(Identifiable.randomUID("PDF_TO_TEXT_TRANSFORMER"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  protected def outputDataType: StructType = new StructType()
    .add($(outputCol), StringType)
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

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  setDefault(inputCol -> "content", outputCol -> "text")

  private def transformUDF: UserDefinedFunction = udf(
    (path: String, content: Array[Byte]) => {
      doProcess(content)
    },
    ArrayType(outputDataType))

  private def doProcess(
      content: Array[Byte]): Seq[(String, Int, Int, Array[Byte], String, Int)] = {
    val pagesTry = Try(
      pdfToText(
        content,
        $(onlyPageNum),
        $(splitPage),
        $(storeSplittedPdf),
        $(sort),
        $(textStripper)))

    pagesTry match {
      case Failure(_) =>
        Seq()
      case Success(content) =>
        content
    }
  }

  override def transform(df: Dataset[_]): DataFrame = {
    transformSchema(df.schema)

    val selCols1 = df.columns
      .filterNot(_ == $(inputCol))
      .map(col) :+ posexplode_outer(transformUDF(df.col($(originCol)), df.col($(inputCol))))
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

      pdfs flatMap { case BinaryFile(bytes, path) =>
        doProcess(bytes).zipWithIndex.map { case ((text, _, _, content, exception, _), pageNum) =>
          val metadata =
            Map("exception" -> exception, "sourcePath" -> path, "pageNum" -> pageNum.toString)

          val result = lightRecord ++ Map(
            getOutputCol -> Seq(OcrText(text, metadata, content)),
            getOrDefault(pageNumCol) -> Seq(PageNum(pageNum)))
          result
        }
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
      textStripper: String): Seq[(String, Int, Int, Array[Byte], String, Int)] = {
    val validPdf = checkAndFixPdf(content)
    val pdfDoc = PDDocument.load(validPdf)
    val numPages = pdfDoc.getNumberOfPages
    log.info(s"Number of pages ${numPages}")
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
        textStripper)
    } else {
      Range(1, numPages + 1).map(pageNum => ("", 1, 1, null, null, pageNum))
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
      textStripper: String): Seq[(String, Int, Int, Array[Byte], String, Int)] = {
    lazy val out: ByteArrayOutputStream = new ByteArrayOutputStream()
    if (splitPage)
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
            (
              text,
              page.getMediaBox.getHeight.toInt,
              page.getMediaBox.getWidth.toInt,
              splittedPdf,
              null,
              pagenum)
          })
    else {
      val text = extractText(pdfDoc, startPage, endPage, sort, textStripper).mkString(
        System.lineSeparator())
      val heightDimension = pdfDoc.getPage(startPage).getMediaBox.getHeight.toInt
      val widthDimension = pdfDoc.getPage(startPage).getMediaBox.getWidth.toInt
      Seq(
        (text, heightDimension, widthDimension, if (storeSplittedPdf) content else null, null, 0))
    }
  }
}

object PdfToText extends DefaultParamsReadable[PdfToText] {
  override def load(path: String): PdfToText = super.load(path)
}
