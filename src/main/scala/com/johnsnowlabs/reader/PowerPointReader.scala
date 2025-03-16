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
import com.johnsnowlabs.reader.util.PptParser
import org.apache.poi.hslf.usermodel.HSLFSlideShow
import org.apache.poi.xslf.usermodel.XMLSlideShow
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import java.io.ByteArrayInputStream
import scala.collection.JavaConverters._

class PowerPointReader(storeContent: Boolean = false) extends Serializable {

  private val spark = ResourceHelper.spark
  import spark.implicits._

  def ppt(filePath: String): DataFrame = {
    if (ResourceHelper.validFile(filePath)) {
      val binaryFilesRDD = spark.sparkContext.binaryFiles(filePath)
      val byteArrayRDD = binaryFilesRDD.map { case (path, portableDataStream) =>
        val byteArray = portableDataStream.toArray()
        (path, byteArray)
      }
      val powerPointDf = byteArrayRDD
        .toDF("path", "content")
        .withColumn("ppt", parsePowerPointUDF(col("content")))
      if (storeContent) powerPointDf.select("path", "ppt", "content")
      else powerPointDf.select("path", "ppt")
    } else throw new IllegalArgumentException(s"Invalid filePath: $filePath")
  }

  private val parsePowerPointUDF = udf((data: Array[Byte]) => {
    parsePowerPoint(data)
  })

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
      PptParser.extractHSLFSlideContent(slide)
    }
    ppt.close()
    elements
  }

  private def parsePptx(slideInputStream: ByteArrayInputStream): Seq[HTMLElement] = {
    val pptx = new XMLSlideShow(slideInputStream)
    val slides = pptx.getSlides

    val elements = slides.asScala.flatMap { slide =>
      PptParser.extractXSLFSlideContent(slide)
    }
    pptx.close()
    elements
  }

}
