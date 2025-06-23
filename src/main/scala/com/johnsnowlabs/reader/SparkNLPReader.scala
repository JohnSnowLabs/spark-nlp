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

import com.johnsnowlabs.nlp.annotators.cleaners.util.CleanerHelper.{
  BLOCK_SPLIT_PATTERN,
  DOUBLE_PARAGRAPH_PATTERN
}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.reader.util.pdf.TextStripperType
import com.johnsnowlabs.reader.util.PartitionOptions.{
  getDefaultBoolean,
  getDefaultInt,
  getDefaultString
}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame

import scala.collection.JavaConverters._

class SparkNLPReader(
    params: java.util.Map[String, String] = new java.util.HashMap(),
    headers: java.util.Map[String, String] = new java.util.HashMap())
    extends Serializable {

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

  private var outputColumn = "reader"

  def setOutputColumn(value: String): Unit = {
    require(value.nonEmpty, "Result column name cannot be empty.")
    outputColumn = value
  }

  def getOutputColumn: String = outputColumn

  def html(htmlPath: String): DataFrame = {
    val htmlReader =
      new HTMLReader(getTitleFontSize, getStoreContent, getTimeout, headers = htmlHeaders)
    setOutputColumn(htmlReader.getOutputColumn)
    htmlReader.read(htmlPath)
  }

  def htmlToHTMLElement(html: String): Seq[HTMLElement] = {
    val htmlReader =
      new HTMLReader(getTitleFontSize, getStoreContent, getTimeout, headers = htmlHeaders)
    setOutputColumn(htmlReader.getOutputColumn)
    htmlReader.htmlToHTMLElement(html)
  }

  def urlToHTMLElement(url: String): Seq[HTMLElement] = {
    val htmlReader =
      new HTMLReader(getTitleFontSize, getStoreContent, getTimeout, headers = htmlHeaders)
    setOutputColumn(htmlReader.getOutputColumn)
    htmlReader.urlToHTMLElement(url)
  }

  def html(urls: Array[String]): DataFrame = {
    val htmlReader =
      new HTMLReader(getTitleFontSize, getStoreContent, getTimeout, headers = htmlHeaders)
    setOutputColumn(htmlReader.getOutputColumn)
    htmlReader.read(urls)
  }

  def html(urls: java.util.List[String]): DataFrame = {
    val htmlReader =
      new HTMLReader(getTitleFontSize, getStoreContent, getTimeout, headers = htmlHeaders)
    setOutputColumn(htmlReader.getOutputColumn)
    htmlReader.read(urls.asScala.toArray)
  }

  private lazy val htmlHeaders: Map[String, String] =
    if (headers == null) Map.empty
    else headers.asScala.toMap.map { case (k, v) => k -> v }

  private def getTitleFontSize: Int = {
    getDefaultInt(params.asScala.toMap, Seq("titleFontSize", "title_font_size"), default = 16)
  }

  private def getStoreContent: Boolean = {
    getDefaultBoolean(params.asScala.toMap, Seq("storeContent", "store_content"), default = false)
  }

  private def getTimeout: Int = {
    getDefaultInt(params.asScala.toMap, Seq("timeout"), default = 30)
  }

  /** Instantiates class to read email files.
    *
    * emailPath: this is a path to a directory of email files or a path to an email file E.g.
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
    setOutputColumn(emailReader.getOutputColumn)
    emailReader.email(emailPath)
  }

  def email(content: Array[Byte]): Seq[HTMLElement] = {
    val emailReader = new EmailReader(getAddAttachmentContent, getStoreContent)
    setOutputColumn(emailReader.getOutputColumn)
    emailReader.emailToHTMLElement(content)
  }

  private def getAddAttachmentContent: Boolean = {
    getDefaultBoolean(
      params.asScala.toMap,
      Seq("addAttachmentContent", "add_attachment_content"),
      default = false)
  }

  /** Instantiates class to read Word files.
    *
    * docPath: this is a path to a directory of Word files or a path to an Word file E.g.
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
    setOutputColumn(wordReader.getOutputColumn)
    wordReader.doc(docPath)
  }

  def doc(content: Array[Byte]): Seq[HTMLElement] = {
    val wordReader = new WordReader(getAddAttachmentContent, getStoreContent)
    setOutputColumn(wordReader.getOutputColumn)
    wordReader.docToHTMLElement(content)
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
      .setExtractCoordinates(getExtractCoordinates)
      .setNormalizeLigatures(getNormalizeLigatures)
    val binaryPdfDF = spark.read.format("binaryFile").load(pdfPath)
    val pipelineModel = new Pipeline()
      .setStages(Array(pdfToText))
      .fit(binaryPdfDF)

    pipelineModel.transform(binaryPdfDF)
  }

  private def getStoreSplittedPdf: Boolean = {
    getDefaultBoolean(
      params.asScala.toMap,
      Seq("storeSplittedPdf", "store_splitted_pdf"),
      default = false)
  }

  private def getSplitPage: Boolean = {
    getDefaultBoolean(params.asScala.toMap, Seq("splitPage", "split_page"), default = true)
  }

  private def getOnlyPageNum: Boolean = {
    getDefaultBoolean(params.asScala.toMap, Seq("onlyPageNum", "only_page_num"), default = false)
  }

  private def getTextStripper: String = {
    getDefaultString(
      params.asScala.toMap,
      Seq("textStripper", "text_stripper"),
      default = TextStripperType.PDF_TEXT_STRIPPER)
  }

  private def getSort: Boolean = {
    getDefaultBoolean(params.asScala.toMap, Seq("sort"), default = false)
  }

  private def getExtractCoordinates: Boolean = {
    getDefaultBoolean(
      params.asScala.toMap,
      Seq("extractCoordinates", "extract_coordinates"),
      default = false)
  }

  private def getNormalizeLigatures: Boolean = {
    getDefaultBoolean(
      params.asScala.toMap,
      Seq("normalizeLigatures", "normalize_ligatures"),
      default = true)
  }

  /** Instantiates class to read Excel files.
    *
    * docPath: this is a path to a directory of Excel files or a path to an Excel file E.g.
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
      new ExcelReader(
        titleFontSize = getTitleFontSize,
        cellSeparator = getCellSeparator,
        storeContent = getStoreContent,
        includePageBreaks = getIncludePageBreaks,
        inferTableStructure = getInferTableStructure,
        appendCells = getAppendCells)
    setOutputColumn(excelReader.getOutputColumn)
    excelReader.xls(docPath)
  }

  def xls(content: Array[Byte]): Seq[HTMLElement] = {
    val excelReader =
      new ExcelReader(
        titleFontSize = getTitleFontSize,
        cellSeparator = getCellSeparator,
        storeContent = getStoreContent,
        includePageBreaks = getIncludePageBreaks,
        inferTableStructure = getInferTableStructure,
        appendCells = getAppendCells)
    setOutputColumn(excelReader.getOutputColumn)
    excelReader.xlsToHTMLElement(content)
  }

  private def getCellSeparator: String = {
    getDefaultString(params.asScala.toMap, Seq("cellSeparator", "cell_separator"), default = "\t")
  }

  private def getInferTableStructure: Boolean = {
    getDefaultBoolean(
      params.asScala.toMap,
      Seq("inferTableStructure", "infer_table_structure"),
      default = false)
  }

  private def getAppendCells: Boolean = {
    getDefaultBoolean(params.asScala.toMap, Seq("appendCells", "append_cells"), default = false)
  }

  /** Instantiates class to read PowerPoint files.
    *
    * docPath: this is a path to a directory of PowerPoint files or a path to an PowerPoint file
    * E.g. "path/power-point/files"
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
    setOutputColumn(powerPointReader.getOutputColumn)
    powerPointReader.ppt(docPath)
  }

  def ppt(content: Array[Byte]): Seq[HTMLElement] = {
    val powerPointReader = new PowerPointReader(getStoreContent)
    setOutputColumn(powerPointReader.getOutputColumn)
    powerPointReader.pptToHTMLElement(content)
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
      getBlockSplit,
      getGroupBrokenParagraphs,
      getParagraphSplit,
      getShortLineWordThreshold,
      getMaxLineCount,
      getThreshold)
    setOutputColumn(textReader.getOutputColumn)
    textReader.txt(filePath)
  }

  def txtToHTMLElement(text: String): Seq[HTMLElement] = {
    val textReader = new TextReader(
      getTitleLengthSize,
      getStoreContent,
      getBlockSplit,
      getGroupBrokenParagraphs,
      getParagraphSplit,
      getShortLineWordThreshold,
      getMaxLineCount,
      getThreshold)
    setOutputColumn(textReader.getOutputColumn)
    textReader.txtToHTMLElement(text)
  }

  def txtContent(content: String): DataFrame = {
    val textReader = new TextReader(
      getTitleLengthSize,
      getStoreContent,
      getBlockSplit,
      getGroupBrokenParagraphs,
      getParagraphSplit,
      getShortLineWordThreshold,
      getMaxLineCount,
      getThreshold)
    textReader.txtContent(content)
  }

  private def getTitleLengthSize: Int = {
    getDefaultInt(params.asScala.toMap, Seq("titleLengthSize", "title_length_size"), default = 50)
  }

  private def getIncludePageBreaks: Boolean = {
    getDefaultBoolean(
      params.asScala.toMap,
      Seq("includePageBreaks", "include_page_breaks"),
      default = false)
  }

  private def getGroupBrokenParagraphs: Boolean = {
    getDefaultBoolean(
      params.asScala.toMap,
      Seq("groupBrokenParagraphs", "group_broken_paragraphs"),
      default = false)
  }

  private def getParagraphSplit: String = {
    getDefaultString(
      params.asScala.toMap,
      Seq("paragraphSplit", "paragraph_split"),
      default = DOUBLE_PARAGRAPH_PATTERN)
  }

  private def getShortLineWordThreshold: Int = {
    getDefaultInt(
      params.asScala.toMap,
      Seq("shortLineWordThreshold", "short_line_word_threshold"),
      default = 5)
  }

  private def getMaxLineCount: Int = {
    getDefaultInt(params.asScala.toMap, Seq("maxLineCount", "max_line_count"), default = 2000)
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

  private def getBlockSplit: String = {
    getDefaultString(
      params.asScala.toMap,
      Seq("blockSplit", "block_split"),
      default = BLOCK_SPLIT_PATTERN)
  }

  /** Instantiates class to read XML files.
    *
    * xmlPath: this is a path to a directory of XML files or a path to an XML file. E.g.,
    * "path/xml/files"
    *
    * ==Example==
    * {{{
    * val xmlPath = "home/user/xml-directory"
    * val sparkNLPReader = new SparkNLPReader()
    * val xmlDf = sparkNLPReader.xml(xmlPath)
    * }}}
    *
    * ==Example 2==
    * You can use SparkNLP for one line of code
    * {{{
    * val xmlDf = SparkNLP.read.xml(xmlPath)
    * }}}
    *
    * {{{
    * xmlDf.select("xml").show(false)
    * +------------------------------------------------------------------------------------------------------------------------+
    * |xml                                                                                                                    |
    * +------------------------------------------------------------------------------------------------------------------------+
    * |[{Title, John Smith, {elementId -> ..., tag -> title}}, {UncategorizedText, Some content..., {elementId -> ...}}]     |
    * +------------------------------------------------------------------------------------------------------------------------+
    *
    * xmlDf.printSchema()
    * root
    *  |-- path: string (nullable = true)
    *  |-- xml: array (nullable = true)
    *  |    |-- element: struct (containsNull = true)
    *  |    |    |-- elementType: string (nullable = true)
    *  |    |    |-- content: string (nullable = true)
    *  |    |    |-- metadata: map (nullable = true)
    *  |    |    |    |-- key: string
    *  |    |    |    |-- value: string (valueContainsNull = true)
    * }}}
    *
    * @param xmlPath
    *   Path to the XML file or directory
    * @return
    *   A DataFrame with parsed XML as structured elements
    */

  def xml(xmlPath: String): DataFrame = {
    val xmlReader = new XMLReader(getStoreContent, getXmlKeepTags, getOnlyLeafNodes)
    xmlReader.read(xmlPath)
  }

  def xmlToHTMLElement(xml: String): Seq[HTMLElement] = {
    val xmlReader = new XMLReader(getStoreContent, getXmlKeepTags, getOnlyLeafNodes)
    xmlReader.parseXml(xml)
  }

  private def getXmlKeepTags: Boolean = {
    getDefaultBoolean(params.asScala.toMap, Seq("xmlKeepTags", "xml_keep_tags"), default = false)
  }

  private def getOnlyLeafNodes: Boolean = {
    getDefaultBoolean(
      params.asScala.toMap,
      Seq("onlyLeafNodes", "only_leaf_nodes"),
      default = true)
  }

  /** Instantiates class to read Markdown (.md) files.
   *
   * This method loads a Markdown file or directory of `.md` files and parses the content into
   * structured elements such as headers, narrative text, lists, and code blocks.
   *
   * ==Example==
   * {{{
   * val mdPath = "home/user/markdown-files"
   * val sparkNLPReader = new SparkNLPReader()
   * val mdDf = sparkNLPReader.md(mdPath)
   * }}}
   *
   * ==Example 2==
   * Use SparkNLP in one line:
   * {{{
   * val mdDf = SparkNLP.read.md(mdPath)
   * }}}
   *
   * {{{
   * mdDf.select("md").show(false)
   *
   * +-----------------------------------------------------------------------------------------------------------------------------------+
   * |md                                                                                                                                 |
   * +-----------------------------------------------------------------------------------------------------------------------------------+
   * |[{Title, Introduction, {level -> 1, paragraph -> 0}}, {NarrativeText, This is a Markdown paragraph., {paragraph -> 0}}, ...]        |
   * +-----------------------------------------------------------------------------------------------------------------------------------+
   *
   * mdDf.printSchema()
   * root
   *  |-- path: string (nullable = true)
   *  |-- md: array (nullable = true)
   *  |    |-- element: struct (containsNull = true)
   *  |    |    |-- elementType: string (nullable = true)
   *  |    |    |-- content: string (nullable = true)
   *  |    |    |-- metadata: map (nullable = true)
   *  |    |    |    |-- key: string
   *  |    |    |    |-- value: string (valueContainsNull = true)
   * }}}
   *
   * @param mdPath
   *   Path to a single .md file or a directory of Markdown files.
   * @return
   *   A DataFrame with parsed Markdown content as structured HTMLElements.
   */
  def md(mdPath: String): DataFrame = {
    val markdownReader = new MarkdownReader()
    setOutputColumn(markdownReader.getOutputColumn)
    markdownReader.md(mdPath)
  }

  def mdToHTMLElement(mdContent: String): Seq[HTMLElement] = {
    val markdownReader = new MarkdownReader()
    setOutputColumn(markdownReader.getOutputColumn)
    markdownReader.parseMarkdown(mdContent)
  }

}
