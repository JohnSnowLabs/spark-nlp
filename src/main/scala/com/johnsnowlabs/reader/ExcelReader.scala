package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.reader.util.XlsxParser.{RichCell, RichRow}
import org.apache.poi.hssf.usermodel.HSSFWorkbook
import org.apache.poi.ss.usermodel.Workbook
import org.apache.poi.xssf.usermodel.XSSFWorkbook
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import java.io.ByteArrayInputStream
import scala.collection.JavaConverters._
import scala.collection.mutable

class ExcelReader(
    titleFontSize: Int = 9,
    cellSeparator: String = "\t",
    storeContent: Boolean = false)
    extends Serializable {

  private val spark = ResourceHelper.spark
  import spark.implicits._

  def xls(filePath: String): DataFrame = {
    if (ResourceHelper.validFile(filePath)) {
      val binaryFilesRDD = spark.sparkContext.binaryFiles(filePath)
      val byteArrayRDD = binaryFilesRDD.map { case (path, portableDataStream) =>
        val byteArray = portableDataStream.toArray()
        (path, byteArray)
      }
      val excelDf = byteArrayRDD
        .toDF("path", "content")
        .withColumn("xls", parseExcelUDF(col("content")))
      if (storeContent) excelDf.select("path", "xls", "content")
      else excelDf.select("path", "xls")
    } else throw new IllegalArgumentException(s"Invalid filePath: $filePath")
  }

  private val parseExcelUDF = udf((data: Array[Byte]) => {
    parseExcel(data)
  })

  // Constants for file type identification
  private val ZipMagicNumberFirstByte: Byte = 0x50.toByte // First byte of ZIP files
  private val ZipMagicNumberSecondByte: Byte = 0x4b.toByte // Second byte of ZIP files
  private val OleMagicNumber: Array[Byte] =
    Array(0xd0.toByte, 0xcf.toByte, 0x11.toByte, 0xe0.toByte) // OLE file header

  private def isXlsxFile(content: Array[Byte]): Boolean = {
    content.length > 1 &&
    content(0) == ZipMagicNumberFirstByte &&
    content(1) == ZipMagicNumberSecondByte
  }

  private def isXlsFile(content: Array[Byte]): Boolean = {
    content.length >= 4 && content.slice(0, 4).sameElements(OleMagicNumber)
  }

  private def parseExcel(content: Array[Byte]): Seq[HTMLElement] = {
    val workbookInputStream = new ByteArrayInputStream(content)
    val workbook: Workbook =
      if (isXlsxFile(content)) new XSSFWorkbook(workbookInputStream)
      else if (isXlsFile(content)) new HSSFWorkbook(workbookInputStream)
      else throw new IllegalArgumentException("Unsupported file format: must be .xls or .xlsx")

    val elementsBuffer = mutable.ArrayBuffer[HTMLElement]()

    for (sheetIndex <- 0 until workbook.getNumberOfSheets) {
      val sheet = workbook.getSheetAt(sheetIndex)
      val sheetName = sheet.getSheetName

      val rowIterator = sheet.iterator()
      while (rowIterator.hasNext) {
        val row = rowIterator.next()
        val elementType =
          if (row.isTitle(titleFontSize)) ElementType.TITLE else ElementType.NARRATIVE_TEXT

        val cellValues = row.cellIterator().asScala.map(_.getCellValue).toSeq
        val content = cellValues.mkString(cellSeparator).trim

        if (content.nonEmpty) {
          val element = HTMLElement(
            elementType = elementType,
            content = content,
            metadata = mutable.Map("SheetName" -> sheetName))
          elementsBuffer += element
        }
      }
    }

    workbook.close()

    elementsBuffer
  }

}
