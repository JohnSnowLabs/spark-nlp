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
import com.johnsnowlabs.partition.util.PartitionHelper.{
  datasetWithTextFile,
  datasetWithTextFileEncoding
}
import com.johnsnowlabs.reader.util.HTMLParser
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.slice
import org.jsoup.nodes.Element

import java.util.regex.Pattern

/** CSVReader partitions CSV files into structured elements with metadata, similar to ExcelReader.
  *
  * @param encoding
  *   Character encoding for reading CSV files (default: UTF-8).
  * @param includeHeader
  *   If true, includes the header as the first row in content and HTML.
  * @param inferTableStructure
  *   If true, generates HTML table representation as metadata.
  */
class CSVReader(
    encoding: String = "UTF-8",
    includeHeader: Boolean = false,
    inferTableStructure: Boolean = true,
    delimiter: String = ",",
    storeContent: Boolean = false,
    outputFormat: String = "json-table")
    extends Serializable {

  private lazy val spark = ResourceHelper.spark

  private var outputColumn = "csv"

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Output column name cannot be empty.")
    outputColumn = value
    this
  }

  def getOutputColumn: String = outputColumn

  /** Main entry: partition CSV files according to chosen strategy. */
  def csv(filePath: String): DataFrame = {
    {
      if (ResourceHelper.validFile(filePath)) {
        val textDf =
          if (encoding.equalsIgnoreCase("utf-8"))
            datasetWithTextFile(spark, filePath)
          else
            datasetWithTextFileEncoding(spark, filePath, encoding)
        val csvDf = buildStructuredCSV(textDf)

        if (storeContent) csvDf.select("path", outputColumn, "content")
        else csvDf.select("path", outputColumn)
      } else {
        throw new IllegalArgumentException(s"Invalid filePath: $filePath")
      }
    }
  }

  def buildStructuredCSV(textDF: DataFrame): DataFrame = {
    import spark.implicits._
    val delimiterPattern = Pattern.quote(delimiter)

    val normalizedDF = textDF.withColumn(
      "lines_array",
      split(regexp_replace(regexp_replace($"content", "\r\n", "\n"), "\r", "\n"), "\n"))

    val linesProcessedDF = if (includeHeader) {
      normalizedDF.withColumn("lines_array_processed", $"lines_array")
    } else {
      normalizedDF.withColumn(
        "lines_array_processed",
        slice($"lines_array", lit(2), size($"lines_array") - 1))
    }

    val nonEmptyLinesDF = linesProcessedDF.withColumn(
      "non_empty_lines",
      filter($"lines_array_processed", x => trim(x) =!= ""))

    val tokensFlattenedDF = nonEmptyLinesDF.withColumn(
      "all_tokens",
      flatten(
        transform(
          col("non_empty_lines"),
          line => filter(transform(split(line, delimiterPattern), trim(_)), t => length(t) > 0))))

    // Reconstruct normalized_content (excluding header if needed)
    val normalizedContentDF =
      tokensFlattenedDF.withColumn("normalized_content", concat_ws(" ", col("all_tokens")))

    if (inferTableStructure) {

      val rowsArrayDF = normalizedContentDF.withColumn(
        "rows_array",
        transform(col("non_empty_lines"), line => split(line, delimiterPattern)))
      val rowsWithTdDF = rowsArrayDF.withColumn(
        "rows_with_td",
        transform(
          $"rows_array",
          row => transform(row, c => concat(lit("<td>"), trim(c), lit("</td>")))))
      val trRowsDF = rowsWithTdDF.withColumn(
        "tr_rows",
        transform($"rows_with_td", row => concat(lit("<tr>"), concat_ws("", row), lit("</tr>"))))
      val htmlTableDF = trRowsDF.withColumn(
        "html_table",
        concat(lit("<table>"), concat_ws("", $"tr_rows"), lit("</table>")))

      outputFormat match {
        case "html-table" =>
          htmlTableDF.withColumn(
            outputColumn,
            array(
              struct(
                lit(ElementType.NARRATIVE_TEXT).as("elementType"),
                $"normalized_content".as("content"),
                map_from_arrays(array(), array()).as("metadata")),
              struct(
                lit(ElementType.TABLE).as("elementType"),
                $"html_table".as("content"),
                map_from_arrays(array(), array()).as("metadata"))))
        case "json-table" =>
          val htmlToJsonUDF = udf { (html: String) =>
            val elem: Element = HTMLParser.parseFirstTableElement(html)
            HTMLParser.tableElementToJson(elem)
          }
          val jsonTableDF = htmlTableDF.withColumn("json_table", htmlToJsonUDF(col("html_table")))
          jsonTableDF.withColumn(
            outputColumn,
            array(
              struct(
                lit(ElementType.NARRATIVE_TEXT).as("elementType"),
                $"normalized_content".as("content"),
                map_from_arrays(array(), array()).as("metadata")),
              struct(
                lit(ElementType.TABLE).as("elementType"),
                $"json_table".as("content"),
                map_from_arrays(array(), array()).as("metadata"))))
        case _ =>
          throw new IllegalArgumentException("Unsupported outputFormat: " + outputFormat)
      }
    } else {
      normalizedContentDF.withColumn(
        outputColumn,
        array(
          struct(
            lit(ElementType.NARRATIVE_TEXT).as("elementType"),
            $"normalized_content".as("content"),
            map_from_arrays(array(), array()).as("metadata"))))

    }
  }

}
