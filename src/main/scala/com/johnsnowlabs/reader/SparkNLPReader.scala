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

import scala.collection.JavaConverters._

class SparkNLPReader(params: java.util.Map[String, String] = new java.util.HashMap()) {

  /** Instantiates class to read HTML files.
    *
    * Two types of input paths are supported,
    *
    * htmlPath: this is a path to a directory of HTML files or a path to an HTML file E.g.
    * "path/html/files"
    *
    * url: this is the URL or set of URLs of a website . E.g., "https://www.wikipedia.org"
    *
    * ==Example==
    * {{{
    * val url = "https://www.wikipedia.org"
    * val sparkNLPReader = new SparkNLPReader()
    * val htmlDf = sparkNLPReader.html(url)
    * }}}
    *
    * ==Example 2==
    * You can use SparkNLP for one line of code
    * {{{
    * val htmlDf = SparkNLP.read.html(url)
    * }}}
    * {{{
    * htmlDf.show(false)
    *
    * +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |url                 |html                                                                                                                                                                                                                                                                                                                            |
    * +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |https://example.com/|[{Title, Example Domain, {pageNumber -> 1}}, {NarrativeText, 0, This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission., {pageNumber -> 1}}, {NarrativeText, 0, More information... More information..., {pageNumber -> 1}}]   |
    * +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    *
    * htmlDf.printSchema()
    * root
    *  |-- url: string (nullable = true)
    *  |-- html: array (nullable = true)
    *  |    |-- element: struct (containsNull = true)
    *  |    |    |-- elementType: string (nullable = true)
    *  |    |    |-- content: string (nullable = true)
    *  |    |    |-- metadata: map (nullable = true)
    *  |    |    |    |-- key: string
    *  |    |    |    |-- value: string (valueContainsNull = true)
    * }}}
    *
    * @param params
    *   Parameter with custom configuration
    */

  def html(htmlPath: String): DataFrame = {
    val htmlReader = new HTMLReader(getTitleFontSize, getStoreContent)
    htmlReader.read(htmlPath)
  }

  def html(urls: Array[String]): DataFrame = {
    val htmlReader = new HTMLReader(getTitleFontSize, getStoreContent)
    htmlReader.read(urls)
  }

  def html(urls: java.util.List[String]): DataFrame = {
    val htmlReader = new HTMLReader(getTitleFontSize, getStoreContent)
    htmlReader.read(urls.asScala.toArray)
  }

  private def getTitleFontSize: Int = {
    val titleFontSize =
      try {
        params.asScala.getOrElse("titleFontSize", "16").toInt
      } catch {
        case _: IllegalArgumentException => 16
      }

    titleFontSize
  }

  private def getStoreContent: Boolean = {
    val storeContent =
      try {
        params.asScala.getOrElse("storeContent", "false").toBoolean
      } catch {
        case _: IllegalArgumentException => false
      }
    storeContent
  }

  /** Instantiates class to read email files.
    *
    * emailPath: this is a path to a directory of HTML files or a path to an HTML file E.g.
    * "path/email/files"
    *
    * ==Example==
    * {{{
    * val emailsPath = "home/user/emails-directory"
    * val sparkNLPReader = new SparkNLPReader()
    * val emailDf = sparkNLPReader.email(emailsPath)
    * }}}
    *
    * ==Example 2==
    * You can use SparkNLP for one line of code
    * {{{
    * val emailDf = SparkNLP.read.email(emailsPath)
    * }}}
    *
    * {{{
    * emailDf.select("email").show(false)
    * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |email                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
    * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |[{Title, Email Text Attachments, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>}}, {NarrativeText, Email  test with two text attachments\r\n\r\nCheers,\r\n\r\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/plain}}, {NarrativeText, <html>\r\n<head>\r\n<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">\r\n<style type="text/css" style="display:none;"> P {margin-top:0;margin-bottom:0;} </style>\r\n</head>\r\n<body dir="ltr">\r\n<span style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">Email&nbsp; test with two text attachments</span>\r\n<div class="elementToProof" style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">\r\n<br>\r\n</div>\r\n<div class="elementToProof" style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">\r\nCheers,</div>\r\n<div class="elementToProof" style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">\r\n<br>\r\n</div>\r\n</body>\r\n</html>\r\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/html}}, {Attachment, filename.txt, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, contentType -> text/plain; name="filename.txt"}}, {NarrativeText, This is the content of the file.\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/plain}}, {Attachment, filename2.txt, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, contentType -> text/plain; name="filename2.txt"}}, {NarrativeText, This is an additional content file.\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/plain}}]|
    * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    *
    * emailDf.printSchema()
    * root
    *  |-- path: string (nullable = true)
    *  |-- content: binary (nullable = true)
    *  |-- email: array (nullable = true)
    *  |    |-- element: struct (containsNull = true)
    *  |    |    |-- elementType: string (nullable = true)
    *  |    |    |-- content: string (nullable = true)
    *  |    |    |-- metadata: map (nullable = true)
    *  |    |    |    |-- key: string
    *  |    |    |    |-- value: string (valueContainsNull = true)
    * }}}
    *
    * @param params
    *   Parameter with custom configuration
    */

  def email(emailPath: String): DataFrame = {
    val emailReader = new EmailReader(getAddAttachmentContent, getStoreContent)
    emailReader.read(emailPath)
  }

  private def getAddAttachmentContent: Boolean = {
    val addAttachmentContent =
      try {
        params.asScala.getOrElse("addAttachmentContent", "false").toBoolean
      } catch {
        case _: IllegalArgumentException => false
      }
    addAttachmentContent
  }

  /** Instantiates class to read Word files.
   *
   * docPath: this is a path to a directory of Word files or a path to an HTML file E.g.
   * "path/word/files"
   *
   * ==Example==
   * {{{
   * val docsPath = "home/user/word-directory"
   * val sparkNLPReader = new SparkNLPReader()
   * val docsDf = sparkNLPReader.email(docsPath)
   * }}}
   *
   * ==Example 2==
   * You can use SparkNLP for one line of code
   * {{{
   * val docsDf = SparkNLP.read.doc(docsPath)
   * }}}
   *
   * {{{
   * docsDf.select("doc").show(false)
   * +----------------------------------------------------------------------------------------------------------------------------------------------------+
   * |doc                                                                                                                                                 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
   * +----------------------------------------------------------------------------------------------------------------------------------------------------+
   * |[{Table, Header Col 1, {}}, {Table, Header Col 2, {}}, {Table, Lorem ipsum, {}}, {Table, A Link example, {}}, {NarrativeText, Dolor sit amet, {}}]  |
   * +----------------------------------------------------------------------------------------------------------------------------------------------------+
   *
   * docsDf.printSchema()
   * root
   *  |-- path: string (nullable = true)
   *  |-- content: binary (nullable = true)
   *  |-- doc: array (nullable = true)
   *  |    |-- element: struct (containsNull = true)
   *  |    |    |-- elementType: string (nullable = true)
   *  |    |    |-- content: string (nullable = true)
   *  |    |    |-- metadata: map (nullable = true)
   *  |    |    |    |-- key: string
   *  |    |    |    |-- value: string (valueContainsNull = true)
   * }}}
   *
   * @param params
   *   Parameter with custom configuration
   */

  def doc(docPath: String): DataFrame = {
    val wordReader = new WordReader(getStoreContent)
    wordReader.doc(docPath)
  }

}
