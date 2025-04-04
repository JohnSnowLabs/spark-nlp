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

import com.johnsnowlabs.nlp.annotators.cleaners.util.CleanerHelper
import com.johnsnowlabs.nlp.annotators.cleaners.util.CleanerHelper.DOUBLE_PARAGRAPH_PATTERN
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.reader.util.pdf.TextStripperType
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame

import scala.collection.JavaConverters._

class SparkNLPReader(
    params: java.util.Map[String, String] = new java.util.HashMap(),
    headers: java.util.Map[String, String] = new java.util.HashMap()) {

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
    val htmlReader = new HTMLReader(
      getTitleFontSize,
      getStoreContent,
      getTimeout,
      headers = headers.asScala.toMap)
    htmlReader.read(htmlPath)
  }

  def html(urls: Array[String]): DataFrame = {
    val htmlReader = new HTMLReader(
      getTitleFontSize,
      getStoreContent,
      getTimeout,
      headers = headers.asScala.toMap)
    htmlReader.read(urls)
  }

  def html(urls: java.util.List[String]): DataFrame = {
    val htmlReader = new HTMLReader(
      getTitleFontSize,
      getStoreContent,
      getTimeout,
      headers = headers.asScala.toMap)
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

  private def getTimeout: Int = {
    val timeout =
      try {
        params.asScala.getOrElse("timeout", "30").toInt
      } catch {
        case _: IllegalArgumentException => 30
      }

    timeout
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
    val wordReader = new WordReader(getStoreContent, getIncludePageBreaks)
    wordReader.doc(docPath)
  }

  /** Instantiates class to read PDF files.
    *
    * pdfPath: this is a path to a directory of PDF files or a path to an PDF file E.g.
    * "path/pdfs/"
    *
    * ==Example==
    * {{{
    * val pdfsPath = "home/user/pdfs-directory"
    * val sparkNLPReader = new SparkNLPReader()
    * val pdfDf = sparkNLPReader.pdf(pdfsPath)
    * }}}
    *
    * ==Example 2==
    * You can use SparkNLP for one line of code
    * {{{
    * val pdfDf = SparkNLP.read.pdf(pdfsPath)
    * }}}
    *
    * {{{
    * pdfDf.show(false)
    * +--------------------+--------------------+------+--------------------+----------------+---------------+--------------------+---------+-------+
    * |                path|    modificationTime|length|                text|height_dimension|width_dimension|             content|exception|pagenum|
    * +--------------------+--------------------+------+--------------------+----------------+---------------+--------------------+---------+-------+
    * |file:/content/pdf...|2025-01-15 20:48:...| 25803|This is a Title \...|             842|            596|[25 50 44 46 2D 3...|     NULL|      0|
    * |file:/content/pdf...|2025-01-15 20:48:...|  9487|This is a page.\n...|             841|            595|[25 50 44 46 2D 3...|     NULL|      0|
    * +--------------------+--------------------+------+--------------------+----------------+---------------+--------------------+---------+-------+
    *
    * pdf_df.printSchema()
    * root
    *  |-- path: string (nullable = true)
    *  |-- modificationTime: timestamp (nullable = true)
    *  |-- length: long (nullable = true)
    *  |-- text: string (nullable = true)
    *  |-- height_dimension: integer (nullable = true)
    *  |-- width_dimension: integer (nullable = true)
    *  |-- content: binary (nullable = true)
    *  |-- exception: string (nullable = true)
    *  |-- pagenum: integer (nullable = true)
    * }}}
    *
    * @param params
    *   Parameter with custom configuration
    */
  def pdf(pdfPath: String): DataFrame = {
    val spark = ResourceHelper.spark
    spark.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")
    val pdfToText = new PdfToText()
      .setStoreSplittedPdf(getStoreSplittedPdf)
      .setSplitPage(getSplitPage)
      .setOnlyPageNum(getOnlyPageNum)
      .setTextStripper(getTextStripper)
      .setSort(getSort)
    val binaryPdfDF = spark.read.format("binaryFile").load(pdfPath)
    val pipelineModel = new Pipeline()
      .setStages(Array(pdfToText))
      .fit(binaryPdfDF)

    pipelineModel.transform(binaryPdfDF)
  }

  private def getStoreSplittedPdf: Boolean = {
    val splitPage =
      try {
        params.asScala.getOrElse("storeSplittedPdf", "false").toBoolean
      } catch {
        case _: IllegalArgumentException => false
      }
    splitPage
  }

  private def getSplitPage: Boolean = {
    val splitPage =
      try {
        params.asScala.getOrElse("splitPage", "true").toBoolean
      } catch {
        case _: IllegalArgumentException => true
      }
    splitPage
  }

  private def getOnlyPageNum: Boolean = {
    val splitPage =
      try {
        params.asScala.getOrElse("onlyPageNum", "false").toBoolean
      } catch {
        case _: IllegalArgumentException => false
      }
    splitPage
  }

  private def getTextStripper: String = {
    val textStripper =
      try {
        params.asScala.getOrElse("textStripper", TextStripperType.PDF_TEXT_STRIPPER)
      } catch {
        case _: IllegalArgumentException => TextStripperType.PDF_TEXT_STRIPPER
      }
    textStripper
  }

  private def getSort: Boolean = {
    val sort =
      try {
        params.asScala.getOrElse("sort", "false").toBoolean
      } catch {
        case _: IllegalArgumentException => false
      }
    sort
  }

  /** Instantiates class to read Excel files.
    *
    * docPath: this is a path to a directory of Excel files or a path to an HTML file E.g.
    * "path/excel/files"
    *
    * ==Example==
    * {{{
    * val docsPath = "home/user/excel-directory"
    * val sparkNLPReader = new SparkNLPReader()
    * val xlsDf = sparkNLPReader.xls(docsPath)
    * }}}
    *
    * ==Example 2==
    * You can use SparkNLP for one line of code
    * {{{
    * val xlsDf = SparkNLP.read.xls(docsPath)
    * }}}
    *
    * {{{
    * xlsDf.select("xls").show(false)
    * +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |xls                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
    * +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |[{Title, Financial performance, {SheetName -> Index}}, {Title, Topic\tPeriod\t\t\tPage, {SheetName -> Index}}, {NarrativeText, Quarterly revenue\tNine quarters to 30 June 2023\t\t\t1.0, {SheetName -> Index}}, {NarrativeText, Group financial performance\tFY 22\tFY 23\t\t2.0, {SheetName -> Index}}, {NarrativeText, Segmental results\tFY 22\tFY 23\t\t3.0, {SheetName -> Index}}, {NarrativeText, Segmental analysis\tFY 22\tFY 23\t\t4.0, {SheetName -> Index}}, {NarrativeText, Cash flow\tFY 22\tFY 23\t\t5.0, {SheetName -> Index}}, {Title, Operational metrics, {SheetName -> Index}}, {Title, Topic\tPeriod\t\t\tPage, {SheetName -> Index}}, {NarrativeText, Mobile customers\tNine quarters to 30 June 2023\t\t\t6.0, {SheetName -> Index}}, {NarrativeText, Fixed broadband customers\tNine quarters to 30 June 2023\t\t\t7.0, {SheetName -> Index}}, {NarrativeText, Marketable homes passed\tNine quarters to 30 June 2023\t\t\t8.0, {SheetName -> Index}}, {NarrativeText, TV customers\tNine quarters to 30 June 2023\t\t\t9.0, {SheetName -> Index}}, {NarrativeText, Converged customers\tNine quarters to 30 June 2023\t\t\t10.0, {SheetName -> Index}}, {NarrativeText, Mobile churn\tNine quarters to 30 June 2023\t\t\t11.0, {SheetName -> Index}}, {NarrativeText, Mobile data usage\tNine quarters to 30 June 2023\t\t\t12.0, {SheetName -> Index}}, {NarrativeText, Mobile ARPU\tNine quarters to 30 June 2023\t\t\t13.0, {SheetName -> Index}}, {Title, Other, {SheetName -> Index}}, {Title, Topic\tPeriod\t\t\tPage, {SheetName -> Index}}, {NarrativeText, Average foreign exchange rates\tNine quarters to 30 June 2023\t\t\t14.0, {SheetName -> Index}}, {NarrativeText, Guidance rates\tFY 23/24\t\t\t14.0, {SheetName -> Index}}]|
    * +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    *
    * xlsDf.printSchema()
    * root
    *  |-- path: string (nullable = true)
    *  |-- content: binary (nullable = true)
    *  |-- xls: array (nullable = true)
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

  def xls(docPath: String): DataFrame = {
    val excelReader =
      new ExcelReader(getTitleFontSize, getCellSeparator, getStoreContent, getIncludePageBreaks)
    excelReader.xls(docPath)
  }

  private def getCellSeparator: String = {
    params.asScala.getOrElse("cellSeparator", "\t")
  }

  /** Instantiates class to read PowerPoint files.
    *
    * docPath: this is a path to a directory of Excel files or a path to an HTML file E.g.
    * "path/power-point/files"
    *
    * ==Example==
    * {{{
    * val docsPath = "home/user/power-point-directory"
    * val sparkNLPReader = new SparkNLPReader()
    * val pptDf = sparkNLPReader.ppt(docsPath)
    * }}}
    *
    * ==Example 2==
    * You can use SparkNLP for one line of code
    * {{{
    * val pptDf = SparkNLP.read.ppt(docsPath)
    * }}}
    *
    * {{{
    * xlsDf.select("ppt").show(false)
    * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |ppt                                                                                                                                                                                                                                                                                                                      |
    * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |[{Title, Adding a Bullet Slide, {}}, {ListItem, • Find the bullet slide layout, {}}, {ListItem, – Use _TextFrame.text for first bullet, {}}, {ListItem, • Use _TextFrame.add_paragraph() for subsequent bullets, {}}, {NarrativeText, Here is a lot of text!, {}}, {NarrativeText, Here is some text in a text box!, {}}]|
    * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    *
    * pptDf.printSchema()
    * root
    *  |-- path: string (nullable = true)
    *  |-- content: binary (nullable = true)
    *  |-- ppt: array (nullable = true)
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

  def ppt(docPath: String): DataFrame = {
    val powerPointReader = new PowerPointReader(getStoreContent)
    powerPointReader.ppt(docPath)
  }

  /** Instantiates class to read txt files.
    *
    * filePath: this is a path to a directory of TXT files or a path to an TXT file E.g.
    * "path/txt/files"
    *
    * ==Example==
    * {{{
    * val filePath = "home/user/txt/files"
    * val sparkNLPReader = new SparkNLPReader()
    * val txtDf = sparkNLPReader.txt(filePath)
    * }}}
    *
    * ==Example 2==
    * You can use SparkNLP for one line of code
    * {{{
    * val txtDf = SparkNLP.read.txt(filePath)
    * }}}
    *
    * {{{
    * txtDf.select("txt").show(false)
    * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |txt                                                                                                                                                                                                                                                                                                                                                                                                                                        |
    * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |[{Title, BIG DATA ANALYTICS, {paragraph -> 0}}, {NarrativeText, Apache Spark is a fast and general-purpose cluster computing system.\nIt provides high-level APIs in Java, Scala, Python, and R., {paragraph -> 0}}, {Title, MACHINE LEARNING, {paragraph -> 1}}, {NarrativeText, Spark's MLlib provides scalable machine learning algorithms.\nIt includes tools for classification, regression, clustering, and more., {paragraph -> 1}}]|
    * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    *
    * emailDf.printSchema()
    * root
    *  |-- path: string (nullable = true)
    *  |-- content: binary (nullable = true)
    *  |-- txt: array (nullable = true)
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
  def txt(filePath: String): DataFrame = {
    val textReader = new TextReader(
      getTitleLengthSize,
      getStoreContent,
      getGroupBrokenParagraphs,
      getParagraphSplit,
      getShortLineWordThreshold,
      getMaxLineCount,
      getThreshold)
    textReader.txt(filePath)
  }

  def txtContent(content: String): DataFrame = {
    val textReader = new TextReader(
      getTitleLengthSize,
      getStoreContent,
      getGroupBrokenParagraphs,
      getParagraphSplit,
      getShortLineWordThreshold,
      getMaxLineCount,
      getThreshold)
    textReader.txtContent(content)
  }

  private def getTitleLengthSize: Int = {
    val titleLengthSize =
      try {
        params.asScala.getOrElse("titleLengthSize", "50").toInt
      } catch {
        case _: IllegalArgumentException => 50
      }

    titleLengthSize
  }

  private def getIncludePageBreaks: Boolean = {
    val includePageBreaks =
      try {
        params.asScala.getOrElse("includePageBreaks", "false").toBoolean
      } catch {
        case _: IllegalArgumentException => false
      }
    includePageBreaks
  }

  private def getGroupBrokenParagraphs: Boolean = {
    val groupBrokenParagraphs =
      try {
        params.asScala.getOrElse("groupBrokenParagraphs", "false").toBoolean
      } catch {
        case _: IllegalArgumentException => false
      }
    groupBrokenParagraphs
  }

  private def getParagraphSplit: String = {
    val paragraphSplit =
      try {
        params.asScala.getOrElse("paragraphSplit", DOUBLE_PARAGRAPH_PATTERN)
      } catch {
        case _: IllegalArgumentException => DOUBLE_PARAGRAPH_PATTERN
      }
    paragraphSplit
  }

  private def getShortLineWordThreshold: Int = {
    val shortLineWordThreshold =
      try {
        params.asScala.getOrElse("shortLineWordThreshold", "5").toInt
      } catch {
        case _: IllegalArgumentException => 5
      }

    shortLineWordThreshold
  }

  private def getMaxLineCount: Int = {
    val maxLineCount =
      try {
        params.asScala.getOrElse("maxLineCount", "2000").toInt
      } catch {
        case _: IllegalArgumentException => 2000
      }

    maxLineCount
  }

  private def getThreshold: Double = {
    val threshold =
      try {
        params.asScala.getOrElse("threshold", "0.1").toDouble
      } catch {
        case _: IllegalArgumentException => 0.1
      }

    threshold
  }

}
