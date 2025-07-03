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
import com.johnsnowlabs.partition.util.PartitionHelper.datasetWithTextFile
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable

class MarkdownReader extends Serializable {

  private lazy val spark: SparkSession = ResourceHelper.spark
  private var outputColumn: String = "md"

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Output column name cannot be empty.")
    outputColumn = value
    this
  }

  def getOutputColumn: String = outputColumn

  def md(filePath: String): DataFrame = {
    if (ResourceHelper.validFile(filePath)) {
      val textDf = datasetWithTextFile(spark, filePath)
        .withColumn(outputColumn, parseMarkdownUDF(col("content")))
      textDf.select("path", outputColumn)
    } else {
      throw new IllegalArgumentException(s"Invalid filePath: $filePath")
    }
  }

  private val parseMarkdownUDF = udf((text: String) => parseMarkdown(text))

  def parseMarkdown(text: String): Seq[HTMLElement] = {
    val lines = text.split("\n")
    val elements = mutable.ArrayBuffer[HTMLElement]()
    var i = 0
    var paragraphIdx = 0
    var inCodeBlock = false
    val codeBuffer = new StringBuilder

    while (i < lines.length) {
      val line = lines(i).trim

      if (line.startsWith("```")) {
        if (inCodeBlock) {
          elements += HTMLElement(
            ElementType.UNCATEGORIZED_TEXT,
            codeBuffer.toString(),
            mutable.Map("paragraph" -> paragraphIdx.toString))
          codeBuffer.clear()
          inCodeBlock = false
          paragraphIdx += 1
        } else {
          inCodeBlock = true
        }
      } else if (inCodeBlock) {
        codeBuffer.append(line).append("\n")
      } else if (line.matches("#{1,6} .*")) {
        val level = line.takeWhile(_ == '#').length
        val content = line.dropWhile(_ == '#').trim
        elements += HTMLElement(
          ElementType.TITLE,
          content,
          mutable.Map("level" -> level.toString, "paragraph" -> paragraphIdx.toString))
        paragraphIdx += 1
      } else if (line.matches("[-*] .*|\\d+\\. .*")) {
        elements += HTMLElement(
          ElementType.LIST_ITEM,
          line,
          mutable.Map("paragraph" -> paragraphIdx.toString))
      } else if (line.nonEmpty) {
        elements += HTMLElement(
          ElementType.NARRATIVE_TEXT,
          line,
          mutable.Map("paragraph" -> paragraphIdx.toString))
      }
      i += 1
    }

    if (inCodeBlock) {
      elements += HTMLElement(
        ElementType.UNCATEGORIZED_TEXT,
        codeBuffer.toString(),
        mutable.Map("paragraph" -> paragraphIdx.toString)
      )
    }

    elements
  }
}
