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
package com.johnsnowlabs.reader

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec
import org.apache.spark.sql.functions.{col, explode}

class PdfReaderTest extends AnyFlatSpec {

  val pdfDirectory = "src/test/resources/reader/pdf/"

  "PdfReader" should "read a PDF file as dataframe" taggedAs FastTest in {
    val pdfReader = new PdfReader()
    val pdfDf = pdfReader.pdf(s"$pdfDirectory/text_3_pages.pdf")
    pdfDf.show()

    assert(!pdfDf.select(col("pdf").getItem(0)).isEmpty)
    assert(!pdfDf.columns.contains("content"))
  }

  it should "store content" taggedAs FastTest in {
    val pdfReader = new PdfReader(storeContent = true)
    val pdfDf = pdfReader.pdf(s"$pdfDirectory/text_3_pages.pdf")
    pdfDf.show()

    assert(!pdfDf.select(col("pdf").getItem(0)).isEmpty)
    assert(pdfDf.columns.contains("content"))
  }

  it should "identify text as titles based on threshold value" taggedAs FastTest in {
    val pdfReader = new PdfReader(titleThreshold = 10)
    val pdfDf = pdfReader.pdf(s"$pdfDirectory/pdf-title.pdf")
    pdfDf.show(false)

    val titleDF = pdfDf
      .select(explode(col("pdf")).as("exploded_pdf"))
      .filter(col("exploded_pdf.elementType") === ElementType.TITLE)
    titleDF.select("exploded_pdf").show(truncate = false)

    assert(titleDF.count() == 3)
  }

  it should "handle corrupted files" taggedAs FastTest in {
    val pdfReader = new PdfReader()
    val pdfDf = pdfReader.pdf(s"src/test/resources/reader/pdf-corrupted/corrupted.pdf")

    val resultDF = pdfDf
      .select(explode(col("pdf")).as("exploded_pdf"))
      .filter(col("exploded_pdf.elementType") === ElementType.UNCATEGORIZED_TEXT)

    assert(resultDF.count() == 1)
  }

}
