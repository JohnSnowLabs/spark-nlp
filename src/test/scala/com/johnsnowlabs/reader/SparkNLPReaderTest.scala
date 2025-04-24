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

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class SparkNLPReaderTest extends AnyFlatSpec {

  "pdf" should "read a PDF file and return a structured Dataframe" taggedAs FastTest in {
    val pdfPath = "src/test/resources/reader/pdf"
    val sparkNLPReader = new SparkNLPReader()
    val pdfDf = sparkNLPReader.pdf(pdfPath)

    assert(pdfDf.count() > 0)
  }

  it should "read a PDF file with params" taggedAs FastTest in {
    val pdfPath = "src/test/resources/reader/pdf"
    val params = new java.util.HashMap[String, String]()
    params.put("storeSplittedPdf", "true")
    val sparkNLPReader = new SparkNLPReader(params)
    val pdfDf = sparkNLPReader.pdf(pdfPath)
    pdfDf.show()

    assert(pdfDf.count() > 0)
  }

}
