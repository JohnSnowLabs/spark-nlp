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

import com.johnsnowlabs.nlp.annotators.cleaners.util.CleanerHelper.{BLOCK_SPLIT_PATTERN, DOUBLE_PARAGRAPH_PATTERN}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.reader.util.TextParser
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import scala.collection.mutable

/** Class to read and parse text files.
  *
  * @param titleLengthSize
  *   Maximum character length used to determine if a text block qualifies as a title during
  *   parsing.
  * @param storeContent
  *   Timeout value in seconds for reading remote HTML resources. Applied when fetching content
  *   from URLs
  * @param groupBrokenParagraphs
  *   Whether to merge fragmented lines into coherent paragraphs using heuristics based on line
  *   length and structure
  * @param paragraphSplit
  *   Regex pattern used to detect paragraph boundaries when grouping broken paragraphs
  * @param shortLineWordThreshold
  *   Maximum word count for a line to be considered 'short' during broken paragraph grouping
  * @param maxLineCount
  *   Maximum number of lines to evaluate when estimating paragraph layout characteristics
  * @param threshold
  *   Threshold ratio of empty lines used to decide between new line-based or broken-paragraph
  *   grouping
  * ==Example==
  * {{{
  * val filePath = "home/user/txt/files"
  * val textReader = new TextReader()
  * val textDf = textReader.txt(filePath)
  * }}}
  *
  * {{{
  * textDf.select("txt").show(false)
  * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |txt                                                                                                                                                                                                                                                                                                                                                                                                                                        |
  * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |[{Title, BIG DATA ANALYTICS, {paragraph -> 0}}, {NarrativeText, Apache Spark is a fast and general-purpose cluster computing system.\nIt provides high-level APIs in Java, Scala, Python, and R., {paragraph -> 0}}, {Title, MACHINE LEARNING, {paragraph -> 1}}, {NarrativeText, Spark's MLlib provides scalable machine learning algorithms.\nIt includes tools for classification, regression, clustering, and more., {paragraph -> 1}}]|
  * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  *
  * textDf.printSchema()
  * root
  *  |-- path: string (nullable = true)
  *  |-- txt: array (nullable = true)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- elementType: string (nullable = true)
  *  |    |    |-- content: string (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  * }}}
  */
class TextReader(
    titleLengthSize: Int = 50,
    storeContent: Boolean = false,
    blockSplit: String = BLOCK_SPLIT_PATTERN,
    groupBrokenParagraphs: Boolean = false,
    paragraphSplit: String = DOUBLE_PARAGRAPH_PATTERN,
    shortLineWordThreshold: Int = 5,
    maxLineCount: Int = 2000,
    threshold: Double = 0.1)
    extends Serializable {

  private lazy val spark = ResourceHelper.spark

  private var outputColumn = "txt"

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Output column name cannot be empty.")
    outputColumn = value
    this
  }

  def getOutputColumn: String = outputColumn

  /** Parses TXT files and returns a DataFrame.
    *
    * The DataFrame will contain:
    *   - "path": the file path,
    *   - "content": the raw text content,
    *   - outputColumn: a Seq[HTMLElement] containing the parsed elements.
    */
  def txt(filePath: String): DataFrame = {
    import spark.implicits._
    if (ResourceHelper.validFile(filePath)) {
      val textFilesRDD = spark.sparkContext.wholeTextFiles(filePath)
      val textDf = textFilesRDD
        .toDF("path", "content")
        .withColumn(outputColumn, parseTxtUDF($"content"))
      if (storeContent) textDf.select("path", outputColumn, "content")
      else textDf.select("path", outputColumn)
    } else {
      throw new IllegalArgumentException(s"Invalid filePath: $filePath")
    }
  }

  def txtContent(content: String): DataFrame = {
    import spark.implicits._
    val df = spark.createDataFrame(Seq(("in-memory", content))).toDF("source", "content")
    val textDf = df.withColumn(outputColumn, parseTxtUDF($"content"))
    if (storeContent) textDf.select(outputColumn, "content")
    else textDf.select(col(outputColumn))
  }

  private val parseTxtUDF = udf((text: String) => parseTxt(text))

  def txtToHTMLElement(text: String): Seq[HTMLElement] = {
    parseTxt(text)
  }

  /** Parses the given text into a sequence of HTMLElements.
    *
    * Parsing logic:
    *   - Split the text into blocks using a delimiter of two or more consecutive newlines.
    *   - Using heuristics, consider a block a title if it is all uppercase and short.
    *   - If a block is a title candidate and the following block exists and is not a title
    *     candidate, treat the first as the Title and the second as its NarrativeText.
    *   - Otherwise, treat blocks as narrative text.
    *   - Omit any element with empty content.
    */
  private def parseTxt(text: String): Seq[HTMLElement] = {
    val processedText = if (groupBrokenParagraphs) {
      TextParser.autoParagraphGrouper(
        text,
        paragraphSplit,
        maxLineCount,
        threshold,
        shortLineWordThreshold)
    } else {
      text
    }

    // Split the processed text into blocks using two or more newlines.
    val blocks = processedText.split(blockSplit).map(_.trim).filter(_.nonEmpty)
    val elements = mutable.ArrayBuffer[HTMLElement]()
    var i = 0
    while (i < blocks.length) {
      val currentBlock = blocks(i)
      if (isTitleCandidate(currentBlock)) {
        elements += HTMLElement(
          ElementType.TITLE,
          currentBlock,
          mutable.Map("paragraph" -> (i / 2).toString))
        if (i + 1 < blocks.length && !isTitleCandidate(blocks(i + 1))) {
          val narrative = blocks(i + 1)
          if (narrative.nonEmpty) {
            elements += HTMLElement(
              ElementType.NARRATIVE_TEXT,
              narrative,
              mutable.Map("paragraph" -> (i / 2).toString))
          }
          i += 2
        } else {
          i += 1
        }
      } else {
        elements += HTMLElement(
          ElementType.NARRATIVE_TEXT,
          currentBlock,
          mutable.Map("paragraph" -> (i / 2).toString))
        i += 1
      }
    }
    elements
  }

  /** Heuristic function to determine if a given line/block is a title candidate.
    *
    * Currently, we consider a block a title candidate if:
    *   - It is non-empty.
    *   - It consists mostly of uppercase letters (ignoring non-letter characters).
    *   - It is relatively short (e.g., 50 characters or fewer).
    */
  private def isTitleCandidate(text: String): Boolean = {
    val trimmed = text.trim
    if (trimmed.isEmpty) return false
    val isAllUpper = trimmed.forall(c => !c.isLetter || c.isUpper)
    val isTitleCase = trimmed.split("\\s+").forall(word => word.headOption.exists(_.isUpper))
    val isShort = trimmed.length <= titleLengthSize
    val hasLetters = trimmed.exists(_.isLetter)
    (isAllUpper || isTitleCase) && isShort && hasLetters
  }

}
