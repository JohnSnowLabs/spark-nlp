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

import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.functions.{col, explode}
import org.scalatest.flatspec.AnyFlatSpec

class ExcelReaderTest extends AnyFlatSpec {

  val docDirectory = "src/test/resources/reader/xls"

  "ExcelReader" should "read an excel file" taggedAs FastTest in {
    val excelReader = new ExcelReader()
    val excelDf = excelReader.xls(s"$docDirectory/2023-half-year-analyses-by-segment.xlsx")
    excelDf.select("xls").show(false)

    assert(!excelDf.select(col("xls").getItem(0)).isEmpty)
    assert(!excelDf.columns.contains("content"))
  }

  "ExcelReader" should "read a directory of excel files" taggedAs FastTest in {
    val excelReader = new ExcelReader()
    val excelDf = excelReader.xls(docDirectory)
    excelDf.select("xls").show(false)

    assert(!excelDf.select(col("xls").getItem(0)).isEmpty)
    assert(!excelDf.columns.contains("content"))
  }

  "ExcelReader" should "read a directory of excel files with custom cell separator" taggedAs FastTest in {
    val excelReader = new ExcelReader(cellSeparator = ";")
    val excelDf = excelReader.xls(s"$docDirectory/vodafone.xlsx")
    excelDf.select("xls").show(false)

    assert(!excelDf.select(col("xls").getItem(0)).isEmpty)
    assert(!excelDf.columns.contains("content"))
  }

  "ExcelReader" should "store content" taggedAs FastTest in {
    val excelReader = new ExcelReader(storeContent = true)
    val excelDf = excelReader.xls(docDirectory)
    excelDf.select("xls").show(false)

    assert(!excelDf.select(col("xls").getItem(0)).isEmpty)
    assert(excelDf.columns.contains("content"))
  }

  it should "work for break pages" taggedAs FastTest in {
    val excelReader = new ExcelReader(includePageBreaks = true)
    val excelDf = excelReader.xls(s"$docDirectory/page-break-example.xlsx")
    excelDf.select("xls").show(false)

    val explodedDf = excelDf.withColumn("xls_exploded", explode(col("xls")))
    val page1Df = explodedDf.filter(
      col("xls_exploded.elementType") === "Title" &&
        col("xls_exploded.content") === "Assets" &&
        col("xls_exploded.metadata")("pageBreak") === "1")
    val page2Df = explodedDf.filter(
      col("xls_exploded.elementType") === "Title" &&
        col("xls_exploded.content") === "Debts" &&
        col("xls_exploded.metadata")("pageBreak") === "2")

    assert(page1Df.count() > 0, "Expected at least one row with Title/Assets and pageBreak = 1")
    assert(page2Df.count() > 0, "Expected at least one row with Title/Debts and pageBreak = 2")
  }

  it should "provide HTML version of the table" taggedAs FastTest in {
    val excelReader = new ExcelReader(inferTableStructure = true)
    val excelDf = excelReader.xls(s"$docDirectory/page-break-example.xlsx")
    val htmlDf = excelDf
      .withColumn("xls_exploded", explode(col("xls")))
      .filter(col("xls_exploded.elementType") === "HTML")
    excelDf.select("xls").show(false)

    assert(!excelDf.select(col("xls").getItem(0)).isEmpty)
    assert(!excelDf.columns.contains("content"))
    assert(htmlDf.count() > 0, "Expected at least one row with HTML element type")
  }

  it should "return all info in one row" taggedAs FastTest in {
    val excelReaderSubtable = new ExcelReader(findSubtable = true)
    val excelSubtableDf = excelReaderSubtable.xls(s"$docDirectory/xlsx-subtable-cases.xlsx")
    val explodedSubtableExcelDf =
      excelSubtableDf.withColumn("xls_exploded", explode(col("xls"))).select("xls_exploded")

    val excelReader = new ExcelReader(findSubtable = false)
    val excelDf = excelReader.xls(s"$docDirectory/xlsx-subtable-cases.xlsx")
    val explodedExcelDf =
      excelDf.withColumn("xls_exploded", explode(col("xls"))).select("xls_exploded")

    explodedSubtableExcelDf.select("xls_exploded").show(false)
    explodedExcelDf.select("xls_exploded").show(false)

    assert(explodedSubtableExcelDf.count() == 1, "Expected only one row with all info")
    assert(explodedExcelDf.count() > 1, "Expected more than one row with all info")
  }

}
