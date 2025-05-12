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
package com.johnsnowlabs.reader.util.pdf.schema

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.pdfbox.text.TextPosition
import org.apache.pdfbox.util.Matrix
import org.apache.pdfbox.pdmodel.font.PDFont
import org.apache.pdfbox.pdmodel.font.PDType1Font

case class MappingMatrix(
    c: String,
    x: Float,
    y: Float,
    width: Float,
    height: Float,
    fontSize: Int,
    source: String) {
  override def toString: String = c

  def toTextPosition(pageWidth: Float, pageHeight: Float, spaceWidth: Int): TextPosition = {
    val DEFAULT_FONT: PDFont = PDType1Font.TIMES_ROMAN
    new TextPosition(
      0,
      pageWidth,
      pageHeight,
      new Matrix(), // DEFAULT_FONT.getFontMatrix,
      this.x + this.width,
      this.y + this.height,
      this.height,
      this.width,
      spaceWidth,
      this.c,
      Array[Int](),
      DEFAULT_FONT,
      this.fontSize,
      this.fontSize)
  }
}

object MappingMatrix {
  val mappingType =
    StructType(
      Seq(
        StructField("c", StringType, true),
        StructField("x", FloatType, false),
        StructField("y", FloatType, false),
        StructField("width", FloatType, false),
        StructField("height", FloatType, false),
        StructField("fontSize", IntegerType, false),
        StructField("source", StringType, true)))

  def fromRow(row: Row): MappingMatrix = {
    MappingMatrix(
      row.getString(0),
      row.getFloat(1),
      row.getFloat(2),
      row.getFloat(3),
      row.getFloat(4),
      row.getInt(5),
      row.getString(6))
  }
}
