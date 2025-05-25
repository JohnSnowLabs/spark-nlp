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
import com.johnsnowlabs.reader.util.PptParser.{RichHSLFSlide, RichXSLFSlide}
import org.apache.poi.hslf.usermodel.HSLFSlideShow
import org.apache.poi.xslf.usermodel.XMLSlideShow
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import java.io.ByteArrayInputStream
import scala.collection.JavaConverters._

/** Class to read and parse PowerPoint files.
  *
  * @param storeContent
  *   Whether to include the raw file content in the output DataFrame as a separate 'content'
  *   column, alongside the structured output. The default value is false.
  * @param inferTableStructure
  *   Whether to generate an HTML table representation from structured table content. When
  *   enabled, a full <table> element is added alongside cell-level elements, based on row and
  *   column layout. The default value is false.
  * @param includeSlideNotes
  *   Whether to extract speaker notes from slides. When enabled, notes are included as narrative
  *   text elements. The default value is false.
  *
  * docPath: this is a path to a directory of Excel files or a path to an HTML file E.g.
  * "path/power-point/files"
  *
  * ==Example==
  * {{{
  * val path = "./ppt-files/fake-power-point.pptx"
  * val powerPointReader = new PowerPointReader()
  * val pptDf = powerPointReader.ppt(path)
  * }}}
  *
  * {{{
  * pptDf.show()
  * +--------------------+--------------------+
  * |                path|                 ppt|
  * +--------------------+--------------------+
  * |file:/content/ppt...|[{Title, Adding a...|
  * +--------------------+--------------------+
  *
  * pptDf.printSchema()
  * root
  *  |-- path: string (nullable = true)
  *  |-- ppt: array (nullable = true)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- elementType: string (nullable = true)
  *  |    |    |-- content: string (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  * }}}
  * For more examples please refer to this
  * [[https://github.com/JohnSnowLabs/spark-nlp/examples/python/reader/SparkNLP_PowerPoint_Reader_Demo.ipynb notebook]].
  */

class PowerPointReader(
    storeContent: Boolean = false,
    inferTableStructure: Boolean = false,
    includeSlideNotes: Boolean = false)
    extends Serializable {

  private lazy val spark = ResourceHelper.spark

  private var outputColumn = "ppt"

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Output column name cannot be empty.")
    outputColumn = value
    this
  }

  def getOutputColumn: String = outputColumn

  /** @param filePath
    *   this is a path to a directory of ppt files or a path to an ppt file E.g. "path/ppts/files"
    *
    * @return
    *   Dataframe with parsed power point content.
    */

  def ppt(filePath: String): DataFrame = {
    if (ResourceHelper.validFile(filePath)) {
      val powerPointDf = datasetWithBinaryFile(spark, filePath)
        .withColumn(outputColumn, parsePowerPointUDF(col("content")))
      if (storeContent) powerPointDf.select("path", outputColumn, "content")
      else powerPointDf.select("path", outputColumn)
    } else throw new IllegalArgumentException(s"Invalid filePath: $filePath")
  }

  private val parsePowerPointUDF = udf((data: Array[Byte]) => {
    parsePowerPoint(data)
  })

  def pptToHTMLElement(content: Array[Byte]): Seq[HTMLElement] = {
    parsePowerPoint(content)
  }

  // Constants for file type identification
  private val ZipMagicNumberFirstByte: Byte = 0x50.toByte // First byte of ZIP files
  private val ZipMagicNumberSecondByte: Byte = 0x4b.toByte // Second byte of ZIP files
  private val OleMagicNumber: Array[Byte] =
    Array(0xd0.toByte, 0xcf.toByte, 0x11.toByte, 0xe0.toByte) // OLE file header

  // Method to check if the file is a .pptx file (ZIP-based)
  private def isPptxFile(content: Array[Byte]): Boolean = {
    content.length > 1 &&
    content(0) == ZipMagicNumberFirstByte &&
    content(1) == ZipMagicNumberSecondByte
  }

  // Method to check if the file is a .ppt file (OLE Compound Document)
  private def isPptFile(content: Array[Byte]): Boolean = {
    content.length >= 4 && content.slice(0, 4).sameElements(OleMagicNumber)
  }

  val titleFontSizeThreshold = 9

  private def parsePowerPoint(content: Array[Byte]): Seq[HTMLElement] = {
    val slideInputStream = new ByteArrayInputStream(content)
    if (isPptxFile(content)) {
      parsePptx(slideInputStream)
    } else if (isPptFile(content)) {
      parsePpt(slideInputStream)
    } else {
      throw new IllegalArgumentException("Unsupported PowerPoint file format")
    }
  }

  private def parsePpt(slideInputStream: ByteArrayInputStream): Seq[HTMLElement] = {
    val ppt = new HSLFSlideShow(slideInputStream)
    val slides = ppt.getSlides

    val elements = slides.asScala.flatMap { slide =>
      slide.extractHSLFSlideContent
    }
    ppt.close()
    elements
  }

  private def parsePptx(slideInputStream: ByteArrayInputStream): Seq[HTMLElement] = {
    val pptx = new XMLSlideShow(slideInputStream)
    val slides = pptx.getSlides

    val elements = slides.asScala.flatMap { slide =>
      slide.extractXSLFSlideContent(inferTableStructure, includeSlideNotes)
    }
    pptx.close()
    elements
  }

}
