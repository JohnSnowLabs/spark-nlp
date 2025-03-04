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
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{col, trim, regexp_replace}
import org.scalatest.flatspec.AnyFlatSpec

class PdfToTextTest extends AnyFlatSpec {

  private val spark = ResourceHelper.spark
  spark.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")

  "PdfToText" should "read PDF files" taggedAs FastTest in {
    val pdfToText = new PdfToText().setStoreSplittedPdf(true)
    val dummyDataFrame = spark.read.format("binaryFile").load("src/test/resources/reader/pdf")
    val pipelineModel = new Pipeline()
      .setStages(Array(pdfToText))
      .fit(dummyDataFrame)

    val pdfDf = pipelineModel.transform(dummyDataFrame)
    pdfDf.show()

    assert(pdfDf.count() > 0)
  }

  it should "not include content data when setStoreSplittedPdf is false" taggedAs FastTest in {
    val pdfToText = new PdfToText()
      .setStoreSplittedPdf(false)
    val dummyDataFrame = spark.read.format("binaryFile").load("src/test/resources/reader/pdf")

    val pipelineModel = new Pipeline()
      .setStages(Array(pdfToText))
      .fit(dummyDataFrame)

    val pdfDf = pipelineModel.transform(dummyDataFrame)
    pdfDf.show()

    assert(pdfDf.filter(col("content").isNotNull).count() == 0)
  }

  it should "identify the correct number of pages" taggedAs FastTest in {
    val pdfToText = new PdfToText()
      .setStoreSplittedPdf(true)
      .setSplitPage(true)
    val dummyDataFrame = spark.read.format("binaryFile").load("src/test/resources/reader/pdf")
    val pipelineModel = new Pipeline()
      .setStages(Array(pdfToText))
      .fit(dummyDataFrame)

    val pdfDf = pipelineModel.transform(dummyDataFrame)
    pdfDf.show()

    assert(pdfDf.filter(col("pagenum") > 0).count() >= 1)
  }

  it should "work with onlyPageNum" taggedAs FastTest in {
    val pdfToText = new PdfToText()
      .setOnlyPageNum(true)
    val dummyDataFrame = spark.read.format("binaryFile").load("src/test/resources/reader/pdf")
    val pipelineModel = new Pipeline()
      .setStages(Array(pdfToText))
      .fit(dummyDataFrame)

    val pdfDf = pipelineModel.transform(dummyDataFrame)
    pdfDf.show(truncate = false)

    assert(pdfDf.filter(col("pagenum") === 0).count() == 0)
    assert(pdfDf.filter(col("text") === "").count() > 0)
  }

  it should "sort a PDF document with scattered text" taggedAs FastTest in {
    import spark.implicits._
    val pdfToText = new PdfToText()
      .setTextStripper("PDFLayoutTextStripper")
      .setSort(true)
    val dummyDataFrame =
      spark.read.format("binaryFile").load("src/test/resources/reader/pdf/unsorted_text.pdf")
    val pipelineModel = new Pipeline()
      .setStages(Array(pdfToText))
      .fit(dummyDataFrame)

    val pdfDf = pipelineModel.transform(dummyDataFrame)
    val actualResult =
      pdfDf.select(trim(regexp_replace(col("text"), "\\s+", " "))).as[String].collect()(0)

    val expectedResult =
      "A random heading up here. Hello, this is line 1. This is line 2, but it's placed above line 3." +
        " Line 3 should be below line 2. Finally, this is line 4, far away."
    assert(actualResult == expectedResult)
  }

  it should "extract coordinates with normalizeLigatures" taggedAs FastTest in {
    val pdfToText = new PdfToText()
      .setStoreSplittedPdf(true)
      .setSplitPage(true)
      .setExtractCoordinates(true)
    val dummyDataFrame =
      spark.read.format("binaryFile").load("src/test/resources/reader/pdf/ligatures_text.pdf")
    val pipelineModel = new Pipeline()
      .setStages(Array(pdfToText))
      .fit(dummyDataFrame)

    val pdfDf = pipelineModel.transform(dummyDataFrame)
    val explodedDF = pdfDf
      .selectExpr("explode(positions) as position")
      .selectExpr("explode(position.mapping) as mapping")
      .select("mapping.c")
    val containsLigature = explodedDF.filter(col("c") === "œ").count() > 0

    assert(!containsLigature)
  }

  it should "extract coordinates without normalizeLigatures" taggedAs FastTest in {
    val pdfToText = new PdfToText()
      .setStoreSplittedPdf(true)
      .setSplitPage(true)
      .setExtractCoordinates(true)
      .setNormalizeLigatures(false)
    val dummyDataFrame =
      spark.read.format("binaryFile").load("src/test/resources/reader/pdf/ligatures_text.pdf")
    val pipelineModel = new Pipeline()
      .setStages(Array(pdfToText))
      .fit(dummyDataFrame)

    val pdfDf = pipelineModel.transform(dummyDataFrame)
    val explodedDF = pdfDf
      .selectExpr("explode(positions) as position")
      .selectExpr("explode(position.mapping) as mapping")
      .select("mapping.c")
    val containsLigature = explodedDF.filter(col("c") === "œ").count() > 0

    assert(containsLigature)
  }

  it should "show exception for corrupted PDF files" in {
    val pdfToText = new PdfToText()
    val dummyDataFrame =
      spark.read.format("binaryFile").load("src/test/resources/reader/pdf-corrupted")
    val pipelineModel = new Pipeline()
      .setStages(Array(pdfToText))
      .fit(dummyDataFrame)

    val pdfDf = pipelineModel.transform(dummyDataFrame)
    pdfDf.select("exception").show(truncate = false)

    assert(pdfDf.filter(col("exception") =!= "").count() > 0)
  }

}
