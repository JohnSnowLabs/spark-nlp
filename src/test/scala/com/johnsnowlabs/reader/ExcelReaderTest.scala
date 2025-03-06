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
    excelDf.select("xls") show (false)

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
