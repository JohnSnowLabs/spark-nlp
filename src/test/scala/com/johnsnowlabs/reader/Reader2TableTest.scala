package com.johnsnowlabs.reader

import com.fasterxml.jackson.databind.ObjectMapper
import com.johnsnowlabs.nlp.AssertAnnotations
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

import scala.util.matching.Regex
import org.apache.spark.sql.functions.{col, size}

class Reader2TableTest extends AnyFlatSpec with SparkSessionTest {

  val htmlFilesDirectory = "src/test/resources/reader/html"
  val excelDirectory = "src/test/resources/reader/xls"
  val wordDirectory = "src/test/resources/reader/doc"
  val pptDirectory = "src/test/resources/reader/ppt"
  val mdDirectory = "src/test/resources/reader/md"
  val csvDirectory = "src/test/resources/reader/csv"
  val unsupportedFiles = "src/test/resources/reader/unsupported-files"

  "Reader2Table" should "convert unstructured input to structured output as JSON" taggedAs FastTest in {

    val reader2Table = new Reader2Table()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/example-caption-th.html")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))
    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    val objectMapper = new ObjectMapper()
    annotationsResult.foreach { annotation =>
      val jsonStringOutput = annotation.head.result
      val resultJson = objectMapper.readTree(jsonStringOutput)

      assert(resultJson.has("caption"), "JSON missing 'caption'")
      assert(resultJson.has("header"), "JSON missing 'header'")
      assert(resultJson.has("rows"), "JSON missing 'rows'")

      val caption = resultJson.get("caption").asText()
      assert(caption.nonEmpty, "Caption should not be empty")

      val header = resultJson.get("header")
      assert(header.isArray, "Header should be an array")
      assert(header.size() > 0, "Header array should not be empty")

      val rows = resultJson.get("rows")
      assert(rows.isArray, "Rows should be an array")
      assert(rows.size() > 0, "Rows array should not be empty")

      rows.forEach { row =>
        assert(row.isArray, "Each row should be an array")
        assert(row.size() > 0, "Row should contain values")
        row.forEach { cell =>
          assert(cell.asText().nonEmpty, "Table cell should not be empty")
        }
      }
    }
  }

  it should "convert unstructured input to structured output as HTML" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/example-caption-th.html")
      .setOutputFormat("html-table")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))
    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    val tableRegex: Regex = "(?i)<table\\b[^>]*>".r // case-insensitive, allows attributes
    annotationsResult.foreach { annotation =>
      val htmlStringOutput = annotation.head.result
      assert(
        tableRegex.findFirstIn(htmlStringOutput).isDefined,
        s"Table HTML content does not contain a valid <table> tag: $htmlStringOutput")
    }
  }

  it should "return only table data" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/example-mix-tags.html")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    annotationsResult.foreach { annotation =>
      val metadata = annotation.head.metadata
      assert(
        metadata.get("elementType").contains(ElementType.TABLE),
        s"Expected elementType 'TABLE', but got: ${metadata.get("elementType")}")
    }
  }

  it should "return empty dataset when there is no table" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/title-test.html")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val emptyCount = resultDf.filter(size(col("document")) === 0).count()
    assert(
      emptyCount == 1,
      s"Expected empty 'document' arrays, but found $emptyCount non-empty rows")
  }

  it should "work with tokenizer" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/example-mix-tags.html")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table, tokenizer))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() == 1)
  }

  it should "merge table content into a single document" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/example-mix-tags.html")
      .setOutputCol("document")
      .setOutputAsDocument(true)
      .setOutputFormat("html-table")

    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")

    annotationsResult.foreach { annotation =>
      val htmlStringOutput = annotation.head.result
      assert(
        htmlStringOutput.contains("<div class=\"tables-group\">"),
        s"Table HTML content does not contain a valid <div> group tag: $htmlStringOutput")
      assert(
        htmlStringOutput.contains("<table>"),
        s"Table HTML content does not contain a valid <table> tag: $htmlStringOutput")
    }
    assert(annotationsResult.head.size == 1)
  }

  it should "output tables for Excel files" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("application/vnd.ms-excel")
      .setContentPath(s"$excelDirectory/simple-example-2tables.xlsx")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    val objectMapper = new ObjectMapper()
    annotationsResult.foreach { annotation =>
      val jsonStringOutput = annotation.head.result
      val resultJson = objectMapper.readTree(jsonStringOutput)

      assert(resultJson.has("caption"), "JSON missing 'caption'")
      assert(resultJson.has("header"), "JSON missing 'header'")
      assert(resultJson.has("rows"), "JSON missing 'rows'")
    }
  }

  it should "output tables as HTML for Excel files" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("application/vnd.ms-excel")
      .setContentPath(s"$excelDirectory/simple-example-2tables.xlsx")
      .setOutputFormat("html-table")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    val tableRegex: Regex = "(?i)<table\\b[^>]*>".r // case-insensitive, allows attributes
    annotationsResult.foreach { annotation =>
      val htmlStringOutput = annotation.head.result
      assert(
        tableRegex.findFirstIn(htmlStringOutput).isDefined,
        s"Table HTML content does not contain a valid <table> tag: $htmlStringOutput")
    }
  }

  it should "output tables for Word files" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("application/msword")
      .setContentPath(s"$wordDirectory/fake_table.docx")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    val objectMapper = new ObjectMapper()
    annotationsResult.foreach { annotation =>
      val jsonStringOutput = annotation.head.result
      val resultJson = objectMapper.readTree(jsonStringOutput)

      assert(resultJson.has("caption"), "JSON missing 'caption'")
      assert(resultJson.has("header"), "JSON missing 'header'")
      assert(resultJson.has("rows"), "JSON missing 'rows'")
    }
  }

  it should "output tables as HTML for Word files" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("application/msword")
      .setContentPath(s"$wordDirectory/fake_table.docx")
      .setOutputFormat("html-table")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    val tableRegex: Regex = "(?i)<table\\b[^>]*>".r // case-insensitive, allows attributes
    annotationsResult.foreach { annotation =>
      val htmlStringOutput = annotation.head.result
      assert(
        tableRegex.findFirstIn(htmlStringOutput).isDefined,
        s"Table HTML content does not contain a valid <table> tag: $htmlStringOutput")
    }
  }

  it should "output tables for PowerPoint files" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("application/vnd.ms-powerpoint")
      .setContentPath(s"$pptDirectory/fake-power-point-table.pptx")
      .setOutputCol("document")
    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    val objectMapper = new ObjectMapper()
    annotationsResult.foreach { annotation =>
      val jsonStringOutput = annotation.head.result
      val resultJson = objectMapper.readTree(jsonStringOutput)

      assert(resultJson.has("caption"), "JSON missing 'caption'")
      assert(resultJson.has("header"), "JSON missing 'header'")
      assert(resultJson.has("rows"), "JSON missing 'rows'")
    }
  }

  it should "output tables as HTML for PowerPoint files" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("application/vnd.ms-powerpoint")
      .setContentPath(s"$pptDirectory/fake-power-point-table.pptx")
      .setOutputFormat("html-table")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    val tableRegex: Regex = "(?i)<table\\b[^>]*>".r // case-insensitive, allows attributes
    annotationsResult.foreach { annotation =>
      val htmlStringOutput = annotation.head.result
      assert(
        tableRegex.findFirstIn(htmlStringOutput).isDefined,
        s"Table HTML content does not contain a valid <table> tag: $htmlStringOutput")
    }
  }

  it should "output tables for Markdown files" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("text/markdown")
      .setContentPath(s"$mdDirectory/simple-table.md")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    val objectMapper = new ObjectMapper()
    annotationsResult.foreach { annotation =>
      val jsonStringOutput = annotation.head.result
      val resultJson = objectMapper.readTree(jsonStringOutput)

      assert(resultJson.has("caption"), "JSON missing 'caption'")
      assert(resultJson.has("header"), "JSON missing 'header'")
      assert(resultJson.has("rows"), "JSON missing 'rows'")
    }
  }

  it should "output tables as HTML for Markdown files" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("text/markdown")
      .setContentPath(s"$mdDirectory/simple-table.md")
      .setOutputFormat("html-table")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    val tableRegex: Regex = "(?i)<table\\b[^>]*>".r // case-insensitive, allows attributes
    annotationsResult.foreach { annotation =>
      val htmlStringOutput = annotation.head.result
      assert(
        tableRegex.findFirstIn(htmlStringOutput).isDefined,
        s"Table HTML content does not contain a valid <table> tag: $htmlStringOutput")
    }
  }

  it should "output tables as JSON for CSV files" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("text/csv")
      .setContentPath(s"$csvDirectory/stanley-cups.csv")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    val objectMapper = new ObjectMapper()
    annotationsResult.foreach { annotation =>
      val jsonStringOutput = annotation.head.result
      val resultJson = objectMapper.readTree(jsonStringOutput)

      assert(resultJson.has("caption"), "JSON missing 'caption'")
      assert(resultJson.has("header"), "JSON missing 'header'")
      assert(resultJson.has("rows"), "JSON missing 'rows'")
    }
  }

  it should "output tables as HTML for CSV files" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentType("text/csv")
      .setContentPath(s"$csvDirectory/stanley-cups.csv")
      .setOutputFormat("html-table")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    val tableRegex: Regex = "(?i)<table\\b[^>]*>".r // case-insensitive, allows attributes
    annotationsResult.foreach { annotation =>
      val htmlStringOutput = annotation.head.result
      assert(
        tableRegex.findFirstIn(htmlStringOutput).isDefined,
        s"Table HTML content does not contain a valid <table> tag: $htmlStringOutput")
    }
  }

  it should "load all files from a directory" taggedAs FastTest in {
    val reader2Table = new Reader2Table()
      .setContentPath("src/test/resources/reader")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))
    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() > 1)
  }

  it should "process unsupported files and display an error in a row without stopping the whole batch" taggedAs SlowTest in {
    val reader2Table = new Reader2Table()
      .setContentPath(unsupportedFiles)
      .setOutputCol("table")
      .setIgnoreExceptions(false)

    val pipeline = new Pipeline().setStages(Array(reader2Table))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    assert(resultDf.filter(col("exception").isNotNull).count() >= 1)
  }

  it should "read from spark dataframe" taggedAs FastTest in {
    val content: String =
      """<html>
        |  <body>
        |    <table>
        |      <tr>
        |        <td>Hello World</td>
        |      </tr>
        |    </table>
        |  </body>
        |</html>""".stripMargin

    val htmlDf = spark.createDataFrame(Seq((1, content))).toDF("id", "html")

    htmlDf.show(truncate = false)

    val reader2Table = new Reader2Table()
      .setInputCol("html")
      .setContentType("text/html")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Table))
    val resultDf = pipeline.fit(htmlDf).transform(htmlDf)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    val objectMapper = new ObjectMapper()
    annotationsResult.foreach { annotation =>
      val jsonStringOutput = annotation.head.result
      val resultJson = objectMapper.readTree(jsonStringOutput)

      assert(resultJson.has("caption"), "JSON missing 'caption'")
      assert(resultJson.has("header"), "JSON missing 'header'")
      assert(resultJson.has("rows"), "JSON missing 'rows'")

      val caption = resultJson.get("caption").asText()
      assert(caption.isEmpty, "Caption should be empty")

      val header = resultJson.get("header")
      assert(header.isArray, "Header should be an array")
      assert(header.size() == 0, "Header array should be empty")

      val rows = resultJson.get("rows")
      assert(rows.isArray, "Rows should be an array")
      assert(rows.size() > 0, "Rows array should not be empty")

      rows.forEach { row =>
        assert(row.isArray, "Each row should be an array")
        assert(row.size() > 0, "Row should contain values")
        row.forEach { cell =>
          assert(cell.asText().nonEmpty, "Table cell should not be empty")
        }
      }
    }
  }

  it should "work for csv files" taggedAs FastTest in {
    val csvDirectory = "src/test/resources/reader/csv"
    val reader2Table = new Reader2Table()
      .setContentType("text/csv")
      .setContentPath(s"$csvDirectory/stanley-cups.csv")
      .setOutputCol("document")
      .setExplodeDocs(true)

    val pipeline = new Pipeline().setStages(Array(reader2Table))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)
    resultDf.show(truncate = false)
    //    assert(resultDf.count() == 1)
  }

}
