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
package com.johnsnowlabs.partition

import com.johnsnowlabs.reader.SparkNLPReader
import org.apache.spark.sql.DataFrame

import java.net.URL
import scala.collection.JavaConverters._

/** The Partition class provides a streamlined interface for interacting with Spark NLP readers.
  * It allows you to extract content from various file formats while providing customization using
  * keyword arguments. File types include Email, Excel, HTML, PPT, Text, Word documents.
  *
  * The Partition class simplifies document handling by automatically detecting the document type
  * based on either the file extension or the provided contentType parameter. It then initializes
  * the appropriate document reader using the sparkNLPReader class, making it convenient to use.
  *
  * @param params
  *   Map of parameters with custom configurations.
  *   It includes the following parameters:
  *
  *   - content_type (All): Override automatic file type detection.
  *   - store_content (All): Include raw file content in the output DataFrame as a separate
  *     'content' column.
  *   - timeout (HTML): Timeout in seconds for fetching remote HTML content.
  *   - title_font_size (HTML, Excel): Minimum font size used to identify titles based on
  *     formatting.
  *   - include_page_breaks (Word, Excel): Whether to tag content with page break metadata.
  *   - group_broken_paragraphs (Text): Whether to merge broken lines into full paragraphs using
  *     heuristics.
  *   - title_length_size (Text): Max character length used to qualify text blocks as titles.
  *   - paragraph_split (Text): Regex to detect paragraph boundaries when grouping lines.
  *   - short_line_word_threshold (Text): Max word count for a line to be considered short.
  *   - threshold (Text): Ratio of empty lines used to switch between newline-based and paragraph
  *     grouping.
  *   - max_line_count (Text): Max lines evaluated when analyzing paragraph structure.
  *   - include_slide_notes (PowerPoint): Whether to include speaker notes from slides as
  *     narrative text.
  *   - infer_table_structure (Word, Excel, PowerPoint): Generate full HTML table structure from
  *     parsed table content.
  *   - append_cells (Excel): Append all rows into a single content block instead of individual
  *     elements.
  *   - cell_separator (Excel): String used to join cell values in a row for text output.
  *   - add_attachment_content (Email): Include text content of plain-text attachments in the
  *     output.
  *   - headers (HTML): This is used when a URL is provided, allowing you to set the necessary
  *     headers for the request.
  *
  * ==Example 1 (Reading Text Files)==
  * {{{
  * val txtDirectory = "/content/txtfiles/reader/txt"
  * val textDf = Partition(Map("content_type" -> "text/plain")).partition(txtDirectory)
  * textDf.show()
  *
  * +--------------------+--------------------+
  * |                path|                 txt|
  * +--------------------+--------------------+
  * |file:/content/txt...|[{Title, BIG DATA...|
  * +--------------------+--------------------+
  * }}}
  *
  * ==Example 2 (Reading Email Files)==
  * {{{
  * emailDirectory = "./email-files/test-several-attachments.eml"
  * partitionDf = Partition(Map("content_type" -> "message/rfc822")).partition(emailDirectory)
  * partitionDf.show()
  * +--------------------+--------------------+
  * |                path|               email|
  * +--------------------+--------------------+
  * |file:/content/ema...|[{Title, Test Sev...|
  * +--------------------+--------------------+
  * }}}
  *
  * ==Example 3 (Reading Webpages)==
  * {{{
  *   val htmlDf = Partition().partition("https://www.wikipedia.org")
  *   htmlDf.show()
  *
  * +--------------------+--------------------+
  * |                 url|                html|
  * +--------------------+--------------------+
  * |https://www.wikip...|[{Title, Wikipedi...|
  * +--------------------+--------------------+
  *
  * }}}
  * *
  */

class Partition(params: java.util.Map[String, String] = new java.util.HashMap()) {

  /** Takes a URL/file/directory path to read and parse it's content.
    *
    * @param path
    *   Path to a file or local directory where all files are stored. Supports URLs and DFS file
    *   systems like databricks, HDFS and Microsoft Fabric OneLake.
    * @param headers
    *   If the path is a URL it sets the necessary headers for the request.
    * @return
    *   DataFrame with parsed file content.
    */
  def partition(
      path: String,
      headers: java.util.Map[String, String] = new java.util.HashMap()): DataFrame = {
    val sparkNLPReader = new SparkNLPReader(params, headers)
    if (isUrl(path)) {
      return sparkNLPReader.html(path)
    }

    val contentTypeOpt = Option(params.get("content_type"))

    val reader = contentTypeOpt match {
      case Some(contentType) => getReaderByContentType(contentType, sparkNLPReader)
      case None => getReaderByExtension(path, sparkNLPReader)
    }

    reader(path)
  }

  private def getReaderByContentType(
      contentType: String,
      sparkNLPReader: SparkNLPReader): String => DataFrame = {
    contentType match {
      case "text/plain" => sparkNLPReader.txt
      case "text/html" => sparkNLPReader.html
      case "message/rfc822" => sparkNLPReader.email
      case "application/msword" |
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document" =>
        sparkNLPReader.doc
      case "application/vnd.ms-excel" |
          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" =>
        sparkNLPReader.xls
      case "application/vnd.ms-powerpoint" |
          "application/vnd.openxmlformats-officedocument.presentationml.presentation" =>
        sparkNLPReader.ppt
      case "application/pdf" => sparkNLPReader.pdf
      case _ => throw new IllegalArgumentException(s"Unsupported content type: $contentType")
    }
  }

  private def getReaderByExtension(
      path: String,
      sparkNLPReader: SparkNLPReader): String => DataFrame = {
    val extension = getFileExtension(path)
    extension match {
      case "txt" => sparkNLPReader.txt
      case "html" | "htm" => sparkNLPReader.html
      case "eml" | "msg" => sparkNLPReader.email
      case "doc" | "docx" => sparkNLPReader.doc
      case "xls" | "xlsx" => sparkNLPReader.xls
      case "ppt" | "pptx" => sparkNLPReader.ppt
      case "pdf" => sparkNLPReader.pdf
      case _ => throw new IllegalArgumentException(s"Unsupported file type: $extension")
    }
  }

  /** Parses multiple URL's.
    *
    * @param urls
    *   list of URL's
    * @param headers
    *   sets the necessary headers for the URL request.
    * @return
    *   DataFrame with parsed url content.
    *
    * ==Example==
    * {{{
    * val htmlDf =
    *      Partition().partitionUrls(Array("https://www.wikipedia.org", "https://example.com/"))
    * htmlDf.show()
    *
    * +--------------------+--------------------+
    * |                 url|                html|
    * +--------------------+--------------------+
    * |https://www.wikip...|[{Title, Wikipedi...|
    * |https://example.com/|[{Title, Example ...|
    * +--------------------+--------------------+
    *
    * htmlDf.printSchema()
    * root
    *   |-- url: string (nullable = true)
    *   |-- html: array (nullable = true)
    *   |    |-- element: struct (containsNull = true)
    *   |    |    |-- elementType: string (nullable = true)
    *   |    |    |-- content: string (nullable = true)
    *   |    |    |-- metadata: map (nullable = true)
    *   |    |    |    |-- key: string
    *   |    |    |    |-- value: string (valueContainsNull = true)
    * }}}
    */

  def partitionUrls(urls: Array[String], headers: Map[String, String] = Map.empty): DataFrame = {
    if (urls.isEmpty) throw new IllegalArgumentException("URL array is empty")
    val sparkNLPReader = new SparkNLPReader(params, headers.asJava)
    sparkNLPReader.html(urls)
  }

  def partitionUrlsJava(
      urls: java.util.List[String],
      headers: java.util.Map[String, String] = new java.util.HashMap()): DataFrame = {
    partitionUrls(urls.asScala.toArray, headers.asScala.toMap)
  }

  /** Parses and reads data from a string.
    *
    * @param text
    *   Text data in the form of a string.
    * @return
    *   DataFrame with parsed text content.
    *
    * ==Example==
    * {{{
    *     val content =
    *       """
    *         |The big brown fox
    *         |was walking down the lane.
    *         |
    *         |At the end of the lane,
    *         |the fox met a bear.
    *         |""".stripMargin
    *
    *     val textDf = Partition(Map("groupBrokenParagraphs" -> "true")).partitionText(content)
    *     textDf.show()
    *
    *     +-----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    *     |txt                                                                                                                                                              |
    *     +-----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    *     |[{NarrativeText, The big brown fox was walking down the lane., {paragraph -> 0}}, {NarrativeText, At the end of the lane, the fox met a bear., {paragraph -> 0}}]|
    *     +-----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    *
    *     textDf.printSchema()
    *     root
    *          |-- txt: array (nullable = true)
    *          |    |-- element: struct (containsNull = true)
    *          |    |    |-- elementType: string (nullable = true)
    *          |    |    |-- content: string (nullable = true)
    *          |    |    |-- metadata: map (nullable = true)
    *          |    |    |    |-- key: string
    *          |    |    |    |-- value: string (valueContainsNull = true)
    *
    * }}}
    */
  def partitionText(text: String): DataFrame = {
    val sparkNLPReader = new SparkNLPReader(params)
    sparkNLPReader.txtContent(text)
  }

  private def getFileExtension(path: String): String = {
    path.split("\\.").lastOption.map(_.toLowerCase).getOrElse("")
  }

  private def isUrl(path: String): Boolean = {
    try {
      val url = new URL(path)
      url.getProtocol == "http" || url.getProtocol == "https"
    } catch {
      case _: Exception => false
    }
  }

}

object Partition {
  def apply(params: Map[String, String] = Map.empty): Partition = {
    new Partition(mapAsJavaMap(params))
  }
}
