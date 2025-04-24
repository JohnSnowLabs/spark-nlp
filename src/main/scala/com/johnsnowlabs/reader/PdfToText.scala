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
import com.johnsnowlabs.reader.util.pdf._
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.text.PDFTextStripper
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, posexplode_outer, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.util.{Failure, Success, Try}

class PdfToText(override val uid: String)
    extends Transformer
    with DefaultParamsWritable
    with HasInputValidator
    with HasInputCol
    with HasOutputCol
    with HasLocalProcess
    with PdfToTextTrait {

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

  final val pageNumCol = new Param[String](this, "pageNumCol", "Page number output column name.")
  final val originCol =
    new Param[String](this, "originCol", "Input column name with original path of file.")
  final val partitionNum = new IntParam(this, "partitionNum", "Number of partitions.")
  final val storeSplittedPdf =
    new BooleanParam(this, "storeSplittedPdf", "Force to store bytes content of splitted pdf.")

  /** @group getParam */
  def setOriginCol(value: String): this.type = set(originCol, value)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group getParam */
  def setPartitionNum(value: Int): this.type = set(partitionNum, value)

  /** @group setParam */
  def setStoreSplittedPdf(value: Boolean): this.type = set(storeSplittedPdf, value)

  setDefault(
    inputCol -> "content",
    outputCol -> "text",
    pageNumCol -> "pagenum",
    originCol -> "path",
    partitionNum -> 0,
    storeSplittedPdf -> false)

  private def transformUDF: UserDefinedFunction = udf(
    (path: String, content: Array[Byte]) => {
      doProcess(content)
    },
    ArrayType(outputDataType))

  private def doProcess(
      content: Array[Byte]): Seq[(String, Int, Int, Array[Byte], String, Int)] = {
    val pagesTry = Try(pdfToText(content, $(storeSplittedPdf)))

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
  private def extractText(document: => PDDocument, startPage: Int, endPage: Int): Seq[String] = {
    val pdfTextStripper = new PDFTextStripper
    pdfTextStripper.setStartPage(startPage + 1)
    pdfTextStripper.setEndPage(endPage + 1)
    Seq(pdfTextStripper.getText(document))
  }

  def pdfToText(
      content: Array[Byte],
      storeSplittedPdf: Boolean): Seq[(String, Int, Int, Array[Byte], String, Int)] = {
    val validPdf = checkAndFixPdf(content)
    val pdfDoc = PDDocument.load(validPdf)
    val numPages = pdfDoc.getNumberOfPages
    log.info(s"Number of pages ${numPages}")
    require(numPages >= 1, "pdf input stream cannot be empty")

    val result = pdfboxMethod(pdfDoc, 0, numPages - 1, content, storeSplittedPdf)
    pdfDoc.close()
    log.info("Close pdf")
    result
  }

  private def pdfboxMethod(
      pdfDoc: => PDDocument,
      startPage: Int,
      endPage: Int,
      content: Array[Byte],
      storeSplittedPdf: Boolean): Seq[(String, Int, Int, Array[Byte], String, Int)] = {
    val text = extractText(pdfDoc, startPage, endPage).mkString(System.lineSeparator())
    val heightDimension = pdfDoc.getPage(startPage).getMediaBox.getHeight.toInt
    val widthDimension = pdfDoc.getPage(startPage).getMediaBox.getWidth.toInt
    Seq((text, heightDimension, widthDimension, if (storeSplittedPdf) content else null, null, 0))
  }
}
