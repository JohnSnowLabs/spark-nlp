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
import org.apache.spark.sql.functions.col
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

  it should "not include content data when setStoreSplittedPdf is false" in {
    val pdfToText = new PdfToText().setStoreSplittedPdf(false)
    val dummyDataFrame = spark.read.format("binaryFile").load("src/test/resources/reader/pdf")

    val pipelineModel = new Pipeline()
      .setStages(Array(pdfToText))
      .fit(dummyDataFrame)

    val pdfDf = pipelineModel.transform(dummyDataFrame)
    pdfDf.show()

    assert(pdfDf.filter(col("content").isNotNull).count() == 0)
  }

}
