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

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class Reader2DocTest extends AnyFlatSpec with SparkSessionTest {

  val htmlFilesDirectory = "src/test/resources/reader/html"
  val docDirectory = "src/test/resources/reader/doc"
  val txtDirectory = "src/test/resources/reader/txt/"
  val pdfDirectory = "src/test/resources/reader/pdf/"
  val mdDirectory = "src/test/resources/reader/md"
  val xmlDirectory = "src/test/resources/reader/xml"
  val unsupportedFiles = "src/test/resources/reader/unsupported-files"

  "Reader2Doc" should "convert unstructured input to structured output for HTML" taggedAs FastTest in {

    val reader2Doc = new Reader2Doc()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/example-div.html")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() == 1)
  }

  it should "output clean flatten text without any structured metadata" taggedAs FastTest in {

    val reader2Doc = new Reader2Doc()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/example-div.html")
      .setOutputCol("document")
      .setFlattenOutput(true)

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val expected: Array[Seq[Annotation]] = Array(
      Seq(
        Annotation(
          "document",
          0,
          26,
          "This Text Is Consider Title",
          Map("sentence" -> "0"),
          Array.emptyFloatArray),
        Annotation(
          "document",
          27,
          92,
          "The text here is consider as narrative text, so it's content data.",
          Map("sentence" -> "1"),
          Array.emptyFloatArray)))

    val actual: Array[Seq[Annotation]] = AssertAnnotations.getActualResult(resultDf, "document")

    AssertAnnotations.assertFields(expected, actual)

    for {
      doc <- actual
      annotation <- doc
    } {
      assert(
        annotation.metadata.keySet == Set("sentence"),
        s"Metadata keys should only be 'sentence', but got: ${annotation.metadata.keySet}")
    }
  }

  it should "convert Reader output to Document and explode documents" taggedAs FastTest in {

    val reader2Doc = new Reader2Doc()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/example-div.html")
      .setOutputCol("document")
      .setExplodeDocs(true)

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() > 1)
  }

  it should "work with Tokenizer" taggedAs FastTest in {
    val reader2Doc = new Reader2Doc()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/fake-html.html")
      .setOutputCol("document")
    val pipeline = new Pipeline().setStages(Array(reader2Doc, tokenizer))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() == 1)
  }

  it should "work for Text documents" taggedAs FastTest in {
    val reader2Doc = new Reader2Doc()
      .setContentType("text/plain")
      .setContentPath(s"$txtDirectory/simple-text.txt")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() == 1)
  }

  it should "work for Word documents" taggedAs FastTest in {
    val reader2Doc = new Reader2Doc()
      .setContentType("application/msword")
      .setContentPath(s"$docDirectory/page-breaks.docx")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)
    resultDf.show()
    assert(resultDf.count() == 1)
  }

  it should "work with PDF documents" taggedAs FastTest in {
    val reader2Doc = new Reader2Doc()
      .setContentType("application/pdf")
      .setContentPath(s"$pdfDirectory/pdf-title.pdf")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() == 1)
  }

  it should "work with Markdown" taggedAs FastTest in {
    val reader2Doc = new Reader2Doc()
      .setContentType("text/markdown")
      .setContentPath(s"$mdDirectory/simple.md")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() == 1)
  }

  it should "work with XML" taggedAs FastTest in {
    val reader2Doc = new Reader2Doc()
      .setContentType("application/xml")
      .setContentPath(s"$xmlDirectory/test.xml")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() == 1)
  }

  it should "throw if contentPath is not set" taggedAs FastTest in {
    val reader2Doc = new Reader2Doc()
      .setContentType("text/html")
      .setOutputCol("document")
    val pipeline = new Pipeline().setStages(Array(reader2Doc))
    val pipelineModel = pipeline.fit(emptyDataSet)

    val ex = intercept[IllegalArgumentException] {
      pipelineModel.transform(emptyDataSet)
    }
    ex.getMessage.contains("contentPath must be set")
  }

  it should "throw if contentPath is empty string" taggedAs FastTest in {
    val reader2Doc = new Reader2Doc()
      .setContentType("text/html")
      .setOutputCol("document")
      .setContentPath("   ")

    val pipeline = new Pipeline().setStages(Array(reader2Doc))
    val pipelineModel = pipeline.fit(emptyDataSet)

    val ex = intercept[IllegalArgumentException] {
      pipelineModel.transform(emptyDataSet)
    }
    ex.getMessage.contains("contentPath must be set")
  }

  it should "return all sentences joined into a single document" in {
    val reader2Doc = new Reader2Doc()
      .setContentType("text/html")
      .setOutputCol("document")
      .setContentPath(s"$htmlFilesDirectory/example-mix-tags.html")
      .setOutputAsDocument(true)

    val pipeline = new Pipeline().setStages(Array(reader2Doc))
    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    assert(annotationsResult.head.size == 1, "Expected one document annotation")
  }

  it should "load all files from a directory" taggedAs FastTest in {
    val reader2Doc = new Reader2Doc()
      .setContentPath("src/test/resources/reader")
      .setOutputCol("document")

    val pipeline = new Pipeline().setStages(Array(reader2Doc))
    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() > 1)
  }

  it should "output text from HTML files without table data" taggedAs FastTest in {
    val reader2Doc = new Reader2Doc()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/fake-html.html")
      .setOutputCol("document")
      .setExcludeNonText(true)

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    annotationsResult.foreach { annotations =>
      val tableData =
        annotations.filter(annotation => annotation.metadata("elementType") == "TABLE")
      assert(tableData.isEmpty)
    }
  }

  it should "output text in one row document from HTML files without table data" taggedAs FastTest in {
    val reader2Doc = new Reader2Doc()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/fake-html.html")
      .setOutputCol("document")
      .setExcludeNonText(true)
      .setOutputAsDocument(true)

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    annotationsResult.foreach { annotations =>
      val tableData = annotations.filter(annotation => annotation.result.contains("Column"))
      assert(tableData.isEmpty)
    }
  }

  it should "output data as a single document" taggedAs FastTest in {
    val reader2Doc = new Reader2Doc()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/fake-html.html")
      .setOutputCol("document")
      .setOutputAsDocument(true)

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    annotationsResult.foreach { annotations =>
      assert(annotations.length == 1)
    }
  }

  it should "ignore non-text data with images" taggedAs SlowTest in {
    val reader2Doc = new Reader2Doc()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/example-images.html")
      .setOutputCol("document")
      .setExcludeNonText(true)

    val pipeline = new Pipeline().setStages(Array(reader2Doc))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualResult(resultDf, "document")
    annotationsResult.foreach { annotations =>
      assert(annotations.head.metadata("elementType") != ElementType.IMAGE)
    }
  }

  it should "validate invalid paths" taggedAs SlowTest in {

    val reader2Doc = new Reader2Doc()
      .setContentPath("src/test/resources/reader/uf2")
      .setOutputCol("document")
      .setIgnoreExceptions(false)

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val errorMessage = intercept[IllegalArgumentException] {
      pipeline.fit(emptyDataSet).transform(emptyDataSet)
    }

    assert(
      errorMessage.getMessage.contains("contentPath must point to a valid file or directory"))
  }

  it should "process unsupported files and display an error in a row without stopping the whole batch" taggedAs SlowTest in {

    val reader2Doc = new Reader2Doc()
      .setContentPath(unsupportedFiles)
      .setOutputCol("document")
      .setIgnoreExceptions(false)

    val pipeline = new Pipeline().setStages(Array(reader2Doc))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    assert(resultDf.filter(col("exception").isNotNull).count() >= 1)
  }

}
