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
import org.apache.spark.sql.functions.col
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

}
