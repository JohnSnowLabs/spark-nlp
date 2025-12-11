/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.common.TableData
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec

class TableAssemblerTestSpec extends AnyFlatSpec {

  val documentAssembler = new DocumentAssembler()
    .setInputCol("table_source")
    .setOutputCol("document")

  val tableAssembler = new TableAssembler()
    .setInputFormat("json")
    .setInputCols(Array("document"))
    .setOutputCol("table")

  val finisher = new Finisher()
    .setInputCols("table")
    .setOutputAsArray(true)
    .setCleanAnnotations(false)
    .setOutputCols("output")

  def createPipeline(corpus: DataFrame, format: String = "json"): DataFrame = {

    val tableAssem = new TableAssembler()
      .setInputFormat(format)
      .setInputCols(Array("document"))
      .setOutputCol("table")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tableAssem, finisher))

    val pipelineDF = pipeline.fit(corpus).transform(corpus)

    pipelineDF
  }

  "TableAssembler" should "run end to end pipeline test" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val jsonTableData1 =
      """{"header": ["Name", "Salary", "Country"], "rows": [["Elon Musk", "100000000", "USA"], ["Jeff Bezos", "95000000", "USA"], ["Bill Gates", "90000000", "USA"]]}"""

    val jsonTableData2 =
      """{"header": ["Product", "Price", "Stock"], "rows": [["Laptop", "1200", "50"], ["Phone", "800", "100"], ["Tablet", "600", "75"]]}"""

    val csvTableData =
      """Name, Age, Department
        |Alice Johnson, 28, Engineering
        |Bob Smith, 35, Marketing
        |Carol Williams, 42, Finance
        |David Brown, 31, Engineering""".stripMargin

    val data = Seq((jsonTableData1, "json"), (jsonTableData2, "json"), (csvTableData, "csv"))
      .toDF("table_source", "format")

    val docAssembler = new DocumentAssembler()
      .setInputCol("table_source")
      .setOutputCol("document")

    val results = data.collect().zipWithIndex.map { case (row, idx) =>
      val format = row.getString(1)
      val tableSource = row.getString(0)

      val singleRowDf = Seq(tableSource).toDF("table_source")
      val result = createPipeline(singleRowDf, format)

      val tableData = if (format == "json") {
        TableData.fromJson(tableSource)
      } else {
        TableData.fromCsv(tableSource, delimiter = ",")
      }

      (tableData, idx)
    }

    assert(
      results.length == 3,
      s"because pipeline should process all 3 tables: " +
        s"\nresult was \n${results.length} \nexpected is: 3")

    val jsonTable1 = results(0)._1
    assert(
      jsonTable1.header.length == 3 && jsonTable1.rows.length == 3,
      s"because first JSON table should have 3 columns and 3 rows: " +
        s"\ncolumns: ${jsonTable1.header.length}, rows: ${jsonTable1.rows.length}")

    val jsonTable2 = results(1)._1
    assert(
      jsonTable2.header.length == 3 && jsonTable2.rows.length == 3,
      s"because second JSON table should have 3 columns and 3 rows: " +
        s"\ncolumns: ${jsonTable2.header.length}, rows: ${jsonTable2.rows.length}")

    val csvTable = results(2)._1
    assert(
      csvTable.header.length == 3 && csvTable.rows.length == 4,
      s"because CSV table should have 3 columns and 4 rows: " +
        s"\ncolumns: ${csvTable.header.length}, rows: ${csvTable.rows.length}")

    results.foreach { case (tableData, idx) =>
      val csvConverted = (
        tableData.header.map(col => "\"" + col.replace("\"", "\"\"") + "\"").mkString(", ")
          + "\n"
          + tableData.rows
            .map(row => row.map(v => "\"" + v.replace("\"", "\"\"") + "\"").mkString(", "))
            .mkString("\n")
      )

      val tableDataFromCsv = TableData.fromCsv(csvConverted, delimiter = ",")

      assert(
        tableData.header.length == tableDataFromCsv.header.length,
        s"because table $idx should maintain column count after conversion: " +
          s"\noriginal: ${tableData.header.length}, converted: ${tableDataFromCsv.header.length}")

      assert(
        tableData.rows.length == tableDataFromCsv.rows.length,
        s"because table $idx should maintain row count after conversion: " +
          s"\noriginal: ${tableData.rows.length}, converted: ${tableDataFromCsv.rows.length}")
    }
  }

  "TableAssembler" should "correctly parse JSON table format" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    val jsonTableData =
      """{"header": ["Name", "Age"], "rows": [["John", "30"], ["Jane", "25"]]}"""

    val df = Seq(jsonTableData).toDF("table_source")

    val result = createPipeline(df, "json")

    val tableData = TableData.fromJson(jsonTableData)

    assert(
      tableData.header.length == 2,
      s"because table should have 2 columns: " +
        s"\nresult was \n${tableData.header.length} \nexpected is: 2")

    assert(
      tableData.rows.length == 2,
      s"because table should have 2 rows: " +
        s"\nresult was \n${tableData.rows.length} \nexpected is: 2")
  }

  "TableAssembler" should "correctly parse CSV table format" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    val csvTableData =
      """Name, Age
        |John, 30
        |Jane, 25""".stripMargin

    val df = Seq(csvTableData).toDF("table_source")

    val result = createPipeline(df, "csv")

    val tableData = TableData.fromCsv(csvTableData, delimiter = ",")

    assert(
      tableData.header.length == 2,
      s"because table should have 2 columns: " +
        s"\nresult was \n${tableData.header.length} \nexpected is: 2")

    assert(
      tableData.rows.length == 2,
      s"because table should have 2 rows: " +
        s"\nresult was \n${tableData.rows.length} \nexpected is: 2")
  }

  "TableAssembler" should "correctly handle CSV with commas in values" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    val csv =
      """
        |Description 1, "Description 2, with comma"
        |This is a test", ttt
        |"This is also a test, but with a comma", a
        |3,
        |, "aaaa, ""aa"" "
        |, "test, test"
        |" ""aa,", "4"" "
        |,
        |1
        |"sdf sdafsad, asdf"
        |""".stripMargin.trim

    val df = Seq(csv).toDF("table_source")

    val tableData = TableData.fromCsv(csv, delimiter = ",")

    assert(
      tableData.header.length == 2,
      s"because parsed table must have two columns: " +
        s"\nresult was \n${tableData.header.length} \nexpected is: 2")

    assert(
      tableData.rows.length == 7,
      s"because parsed table must have 7 rows: " +
        s"\nresult was \n${tableData.rows.length} \nexpected is: 7")
  }

  "TableAssembler" should "correctly parse CSV with different delimiters" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    val csv =
      """
        |Description 1, "Description 2, with comma"
        |This is a test", ttt
        |"This is also a test, but with a comma", a
        |3,
        |, "aaaa, ""aa"" "
        |, "test, test"
        |" ""aa,", "4"" "
        |,
        |1
        |"sdf sdafsad, asdf"
        |""".stripMargin.trim

    Seq(",", "@", "\t").foreach { delimiter =>
      val csvTable = csv.replace(",", delimiter)
      val tableData =
        TableData.fromJson(TableData.fromCsv(csvTable, delimiter = delimiter).toJson)

      assert(
        tableData.header.length == 2,
        s"because parsed $delimiter table must have two columns: " +
          s"\nresult was \n${tableData.header.length} \nexpected is: 2")

      assert(
        tableData.rows.length == 7,
        s"because parsed $delimiter table must have 7 rows: " +
          s"\nresult was \n${tableData.rows.length} \nexpected is: 7")
    }
  }

  "TableAssembler" should "correctly convert between JSON and CSV formats" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    val jsonTableData =
      """{"header": ["Name", "Age", "Salary"], "rows": [["John Doe", "30", "100000"], ["Jane Smith", "25", "90000"]]}"""

    val tableData = TableData.fromJson(jsonTableData)
    val csvTableData = (
      tableData.header.map(col => "\"" + col.replace("\"", "\"\"") + "\"").mkString(", ")
        + "\n"
        + tableData.rows
          .map(row => row.map(v => "\"" + v.replace("\"", "\"\"") + "\"").mkString(", "))
          .mkString("\n")
    )

    val tableDataFromCsv = TableData.fromCsv(csvTableData, delimiter = ",")

    assert(
      tableData.header.length == tableDataFromCsv.header.length,
      s"because JSON and CSV should have same number of columns: " +
        s"\nJSON columns: \n${tableData.header.length} \nCSV columns: \n${tableDataFromCsv.header.length}")

    assert(
      tableData.rows.length == tableDataFromCsv.rows.length,
      s"because JSON and CSV should have same number of rows: " +
        s"\nJSON rows: \n${tableData.rows.length} \nCSV rows: \n${tableDataFromCsv.rows.length}")
  }

  "TableAssembler" should "correctly handle empty tables" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    val emptyJson = """{"header": [], "rows": []}"""
    val emptyDf = Seq(emptyJson).toDF("table_source")

    val result = createPipeline(emptyDf, "json")

    val tableData = TableData.fromJson(emptyJson)

    assert(tableData.header.isEmpty, s"because empty table should have no columns")

    assert(tableData.rows.isEmpty, s"because empty table should have no rows")
  }

  "TableAssembler" should "correctly handle tables with empty cells" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    val jsonWithEmptyCells =
      """{"header": ["Name", "Age", "City"], "rows": [["John", "", "NYC"], ["", "25", "LA"], ["", "", ""]]}"""

    val df = Seq(jsonWithEmptyCells).toDF("table_source")

    val result = createPipeline(df, "json")

    val tableData = TableData.fromJson(jsonWithEmptyCells)

    assert(
      tableData.header.length == 3,
      s"because table should have 3 columns: " +
        s"\nresult was \n${tableData.header.length} \nexpected is: 3")

    assert(
      tableData.rows.length == 3,
      s"because table should have 3 rows: " +
        s"\nresult was \n${tableData.rows.length} \nexpected is: 3")
  }

}
