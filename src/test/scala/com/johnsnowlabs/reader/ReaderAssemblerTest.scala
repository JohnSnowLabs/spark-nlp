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

import com.johnsnowlabs.nlp.AssertAnnotations
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.annotators.cv.Qwen2VLTransformer
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class ReaderAssemblerTest extends AnyFlatSpec with SparkSessionTest {

  val filesDirectory = "src/test/resources/reader/"
  val htmlFilesDirectory = "src/test/resources/reader/html"
  val docDirectory = "src/test/resources/reader/doc"
  val csvDirectory = "src/test/resources/reader/csv"
  val pdfDirectory = "src/test/resources/reader/pdf/"

  "ReaderAssembler" should "read HTML files" taggedAs FastTest in {
    val reader = new ReaderAssembler()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/table-image.html")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    val textResult = AssertAnnotations.getActualResult(resultDf, "document_text")
    val tableResult = AssertAnnotations.getActualResult(resultDf, "document_table")
    val imageResult = AssertAnnotations.getActualImageResult(resultDf, "document_image")
    val actualText = textResult.filter(annotation => annotation.nonEmpty)
    val actualTable = tableResult.filter(annotation => annotation.nonEmpty)
    val actualImages = imageResult.filter(annotation => annotation.nonEmpty)

    assert(actualText.nonEmpty)
    assert(actualTable.nonEmpty)
    assert(actualImages.nonEmpty)
  }

  it should "work for word documents" taggedAs FastTest in {
    val reader = new ReaderAssembler()
      .setContentType("application/msword")
      .setContentPath(s"$docDirectory/doc-img-table.docx")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    val textResult = AssertAnnotations.getActualResult(resultDf, "document_text")
    val tableResult = AssertAnnotations.getActualResult(resultDf, "document_table")
    val imageResult = AssertAnnotations.getActualImageResult(resultDf, "document_image")
    val actualText = textResult.filter(annotation => annotation.nonEmpty)
    val actualTable = tableResult.filter(annotation => annotation.nonEmpty)
    val actualImages = imageResult.filter(annotation => annotation.nonEmpty)

    assert(actualText.nonEmpty)
    assert(actualTable.nonEmpty)
    assert(actualImages.nonEmpty)
  }

  it should "work for csv files" taggedAs FastTest in {
    val reader = new ReaderAssembler()
      .setContentType("text/csv")
      .setContentPath(s"$csvDirectory/stanley-cups.csv")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    val textResult = AssertAnnotations.getActualResult(resultDf, "document_text")
    val tableResult = AssertAnnotations.getActualResult(resultDf, "document_table")
    val imageResult = AssertAnnotations.getActualImageResult(resultDf, "document_image")
    val actualText = textResult.filter(annotation => annotation.nonEmpty)
    val actualTable = tableResult.filter(annotation => annotation.nonEmpty)
    val actualImages = imageResult.filter(annotation => annotation.nonEmpty)

    assert(actualText.nonEmpty)
    assert(actualTable.nonEmpty)
    assert(actualImages.isEmpty)
  }

  it should "integrate HTML files with VLM models" taggedAs SlowTest in {
    val reader = new ReaderAssembler()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/table-image.html")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader))
    val readerDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)
    readerDf.show()
    readerDf.printSchema()

    val visualQAClassifier = Qwen2VLTransformer
      .pretrained()
      .setInputCols("document_image")
      .setOutputCol("answer")

    val vlmPipeline = new Pipeline().setStages(Array(visualQAClassifier))
    val resultDf = vlmPipeline.fit(readerDf).transform(readerDf)

    resultDf.select("document_image.origin", "answer.result").show(truncate = false)

  }

  it should "integrate Word files with VLM models" taggedAs SlowTest in {
    val reader = new ReaderAssembler()
      .setContentType("application/msword")
      .setContentPath(s"$docDirectory/doc-img-table.docx")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader))
    val readerDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)
    readerDf.show()
    readerDf.printSchema()

    val visualQAClassifier = Qwen2VLTransformer
      .pretrained()
      .setInputCols("document_image")
      .setOutputCol("answer")

    val vlmPipeline = new Pipeline().setStages(Array(visualQAClassifier))
    val resultDf = vlmPipeline.fit(readerDf).transform(readerDf)

    resultDf.select("document_image.origin", "answer.result").show(truncate = false)
  }

  it should "read PDF files" taggedAs FastTest in {
    val reader = new ReaderAssembler()
      .setContentType("application/pdf")
      .setContentPath(s"$pdfDirectory/pdf-with-2images.pdf")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    val textResult = AssertAnnotations.getActualResult(resultDf, "document_text")
    val tableResult = AssertAnnotations.getActualResult(resultDf, "document_table")
    val imageResult = AssertAnnotations.getActualImageResult(resultDf, "document_image")
    val actualText = textResult.filter(annotation => annotation.nonEmpty)
    val actualTable = tableResult.filter(annotation => annotation.nonEmpty)
    val actualImages = imageResult.filter(annotation => annotation.nonEmpty)

    assert(actualText.nonEmpty)
    assert(actualTable.isEmpty)
    assert(actualImages.nonEmpty)
  }

  it should "integrate PDF files with VLM models" taggedAs SlowTest in {
    val reader = new ReaderAssembler()
      .setContentType("application/pdf")
      .setContentPath(s"$pdfDirectory/pdf-with-2images.pdf")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader))
    val readerDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)
    readerDf.show()
    readerDf.printSchema()

    val visualQAClassifier = Qwen2VLTransformer
      .pretrained()
      .setInputCols("document_image")
      .setOutputCol("answer")

    val vlmPipeline = new Pipeline().setStages(Array(visualQAClassifier))
    val resultDf = vlmPipeline.fit(readerDf).transform(readerDf)

    resultDf.select("document_image.origin", "answer.result").show(truncate = false)
  }

  it should "read from a directory" taggedAs FastTest in {
    val reader = new ReaderAssembler()
      .setContentPath(s"$filesDirectory")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    val textResult = AssertAnnotations.getActualResult(resultDf, "document_text")
    val tableResult = AssertAnnotations.getActualResult(resultDf, "document_table")
    val imageResult = AssertAnnotations.getActualImageResult(resultDf, "document_image")
    val actualText = textResult.filter(annotation => annotation.nonEmpty)
    val actualTable = tableResult.filter(annotation => annotation.nonEmpty)
    val actualImages = imageResult.filter(annotation => annotation.nonEmpty)

    assert(actualText.nonEmpty)
    assert(actualTable.nonEmpty)
    assert(actualImages.nonEmpty)
  }

  it should "work for inputCol" taggedAs FastTest in {

    val content: String =
      """<html>
        |  <body>
        |  <p style="font-size:12pt;">This is a normal paragraph.</p>
        |    <table>
        |      <tr>
        |        <td>Hello World</td>
        |      </tr>
        |    </table>
        |    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA
        |    AAAFCAYAAACNbyblAAAAHElEQVQI12P4
        |    //8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
        |    alt="Red dot" width="5" height="5">
        |  </body>
        |</html>""".stripMargin

    val htmlDf = spark.createDataFrame(Seq((1, content))).toDF("id", "html")

    val reader = new ReaderAssembler()
      .setInputCol("html")
      .setContentType("text/html")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader))
    val resultDf = pipeline.fit(htmlDf).transform(htmlDf)

    val textResult = AssertAnnotations.getActualResult(resultDf, "document_text")
    val tableResult = AssertAnnotations.getActualResult(resultDf, "document_table")
    val imageResult = AssertAnnotations.getActualImageResult(resultDf, "document_image")
    val actualText = textResult.filter(annotation => annotation.nonEmpty)
    val actualTable = tableResult.filter(annotation => annotation.nonEmpty)
    val actualImages = imageResult.filter(annotation => annotation.nonEmpty)

    assert(actualText.nonEmpty)
    assert(actualTable.nonEmpty)
    assert(actualImages.nonEmpty)
  }

}
