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

import org.apache.spark.sql.DataFrame

import java.util
import scala.collection.JavaConverters._

class SparkNLPReader(params: java.util.Map[String, String] = new util.HashMap()) {

  /** Instantiates class to read HTML files.
   *
   *
   * Two types of input paths are supported,
   *
   * htmlPath: this is a path to a directory of HTML files or a path to an HTML file
   * E.g. "path/html/files"
   *
   * url: this is the URL or set of URLs of a website  . E.g., "https://www.wikipedia.org"
   *
   * ==Example==
   * {{{
   * val url = "https://www.wikipedia.org"
   * val sparkNLPReader = new SparkNLPReader()
   * val htmlDf = sparkNLPReader.html(url)
   * htmlDf.show(false)
   *
   * +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   * |url                 |html                                                                                                                                                                                                                                                                                                                            |
   * +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   * |https://example.com/|[{Title, 0, Example Domain, {pageNumber -> 1}}, {NarrativeText, 0, This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission., {pageNumber -> 1}}, {NarrativeText, 0, More information... More information..., {pageNumber -> 1}}]|
   * +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   *
   * htmlDf.printSchema()
   * root
   *  |-- url: string (nullable = true)
   *   |-- html: array (nullable = true)
   *  |    |-- element: struct (containsNull = true)
   *  |    |    |-- elementType: string (nullable = true)
   *  |    |    |-- elementId: integer (nullable = false)
   *  |    |    |-- content: string (nullable = true)
   *  |    |    |-- metadata: map (nullable = true)
   *  |    |    |    |-- key: string
   *  |    |    |    |-- value: string (valueContainsNull = true)
   * }}}
   *
   * You can use SparkNLP for one line of code
   * ==Example 2==
   * {{{
   * val htmlDf = SparkNLP.read.html(url)
   * htmlDf.show(false)
   *
   * +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   * |url                 |html                                                                                                                                                                                                                                                                                                                            |
   * +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   * |https://example.com/|[{Title, 0, Example Domain, {pageNumber -> 1}}, {NarrativeText, 0, This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission., {pageNumber -> 1}}, {NarrativeText, 0, More information... More information..., {pageNumber -> 1}}]|
   * +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   *
   * }}}
   *
   * @param params
   *   Parameter with custom configuration
   */

  def html(htmlPath: String): DataFrame = {
    val titleFontSize = params.asScala.getOrElse("titleFontSize", "16")
    val htmlReader = new HTMLReader(titleFontSize.toInt)
    htmlReader.read(htmlPath)
  }

  def html(urls: Array[String]): DataFrame = {
    val titleFontSize = params.asScala.getOrElse("titleFontSize", "16")
    val htmlReader = new HTMLReader(titleFontSize.toInt)
    htmlReader.read(urls)
  }

  def html(urls: java.util.List[String]): DataFrame = {
    val titleFontSize = params.asScala.getOrElse("titleFontSize", "16")
    val htmlReader = new HTMLReader(titleFontSize.toInt)
    htmlReader.read(urls.asScala.toArray)
  }

}
