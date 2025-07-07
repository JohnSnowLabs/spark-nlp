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
import com.vladsch.flexmark.ext.tables.TablesExtension
import com.vladsch.flexmark.html.HtmlRenderer
import com.vladsch.flexmark.parser.Parser
import com.vladsch.flexmark.util.data.MutableDataSet
import com.vladsch.flexmark.util.misc.Extension
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.net.{HttpURLConnection, URL}
import java.util.Collections

class MarkdownReader extends Serializable {

  private lazy val spark: SparkSession = ResourceHelper.spark
  private var outputColumn: String = "md"

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Output column name cannot be empty.")
    outputColumn = value
    this
  }

  def getOutputColumn: String = outputColumn

  /** Main entrypoint: supports either filePath or direct text input. Output is always a DataFrame
    * with source and parsed markdown column.
    */
  def md(filePath: String = null, text: String = null): DataFrame = {
    if (filePath != null && filePath.trim.nonEmpty) {
      mdFromFile(filePath)
    } else {
      mdFromText(text)
    }
  }

  /** Reads and parses markdown file from a file path. */
  private def mdFromFile(filePath: String): DataFrame = {
    if (ResourceHelper.validFile(filePath)) {
      val textDf = datasetWithTextFile(spark, filePath)
        .withColumn(outputColumn, parseMarkdownUDF(col("content")))
      textDf.select(col("path").as("source"), col(outputColumn))
    } else {
      throw new IllegalArgumentException(s"Invalid filePath: $filePath")
    }
  }

  /** Parses markdown text directly from a string input. */
  def mdFromText(text: String, source: String = "in-memory"): DataFrame = {
    import spark.implicits._
    val mdDf = spark.createDataFrame(Seq((source, text))).toDF("source", "content")
    val textDf = mdDf.withColumn(outputColumn, parseMarkdownUDF($"content"))
    textDf.select($"source", col(outputColumn))
  }

  def mdFromUrl(url: String): DataFrame = {
    if (!ResourceHelper.isValidURL(url)) {
      throw new IllegalArgumentException(s"Invalid URL: $url")
    }

    val connection = new URL(url).openConnection().asInstanceOf[HttpURLConnection]
    connection.setRequestProperty("Accept", "text/markdown")
    connection.connect()
    val contentType = Option(connection.getContentType).getOrElse("")
    val lowerContentType = contentType.toLowerCase
    if (!lowerContentType.startsWith("text/markdown") && !lowerContentType.startsWith(
        "text/plain"))
      throw new IllegalArgumentException(
        s"Expected Content-Type text/markdown or text/plain, got: $contentType")

    val source = scala.io.Source.fromInputStream(connection.getInputStream, "UTF-8")
    val content =
      try source.mkString
      finally source.close()
    connection.disconnect()

    mdFromText(content, url)
  }

  private val parseMarkdownUDF = udf((text: String) => parseMarkdownWithTables(text))

  def parseMarkdownWithTables(markdown: String): Seq[HTMLElement] = {
    val options = new MutableDataSet()
    val extensions: java.util.List[Extension] = Collections.singletonList(TablesExtension.create())
    options.set(Parser.EXTENSIONS, extensions)

    val parser = Parser.builder(options).build()
    val renderer = HtmlRenderer.builder(options).build()
    val document = parser.parse(markdown)
    val html = renderer.render(document)
    new HTMLReader().htmlToHTMLElement(html).toSeq
  }
}
