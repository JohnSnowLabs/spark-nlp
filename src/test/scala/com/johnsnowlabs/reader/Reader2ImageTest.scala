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

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.annotators.cv.{Qwen2VLTransformer, SmolVLMTransformer}
import com.johnsnowlabs.nlp.{AnnotatorType, AssertAnnotations}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.col
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File

class Reader2ImageTest extends AnyFlatSpec with SparkSessionTest {

  val htmlFilesDirectory = "./src/test/resources/reader/html/"
  val mdDirectory = "src/test/resources/reader/md"
  val mixDirectory = "src/test/resources/reader/mix-files"
  val unsupportedFiles = "src/test/resources/reader/unsupported-files"
  val emailDirectory = "src/test/resources/reader/email/"
  val wordDirectory = "src/test/resources/reader/doc/"
  val imageDirectory = "src/test/resources/reader/img/"
  val pdfDirectory = "src/test/resources/reader/pdf/"
  val filesDirectory = "src/test/resources/reader/"

  "Reader2Image" should "read different image source content from an HTML file" taggedAs SlowTest in {
    val sourceFile = "example-images.html"
    val reader2Image = new Reader2Image()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/$sourceFile")
      .setOutputCol("image")

    val pipeline = new Pipeline().setStages(Array(reader2Image))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)
    resultDf.show()
    val annotationsResult = AssertAnnotations.getActualImageResult(resultDf, "image")

    assert(annotationsResult.length == 2)
    annotationsResult.foreach { annotations =>
      assert(annotations.head.annotatorType == AnnotatorType.IMAGE)
      assert(annotations.head.origin == sourceFile)
      assert(annotations.head.result.nonEmpty)
      assert(annotations.head.height > 0)
      assert(annotations.head.width > 0)
      assert(annotations.head.nChannels > 0)
      assert(annotations.head.mode > 0)
      assert(annotations.head.metadata.nonEmpty)
    }

  }

  it should "read image from a Markdown file" taggedAs SlowTest in {
    val sourceFile = "example-images.md"
    val reader2Image = new Reader2Image()
      .setContentType("text/markdown")
      .setContentPath(s"$mdDirectory/$sourceFile")
      .setOutputCol("image")

    val pipeline = new Pipeline().setStages(Array(reader2Image))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualImageResult(resultDf, "image")

    assert(annotationsResult.length == 2)
    annotationsResult.foreach { annotations =>
      assert(annotations.head.annotatorType == AnnotatorType.IMAGE)
      assert(annotations.head.origin == sourceFile)
      assert(annotations.head.result.nonEmpty)
      assert(annotations.head.height > 0)
      assert(annotations.head.width > 0)
      assert(annotations.head.nChannels > 0)
      assert(annotations.head.mode > 0)
      assert(annotations.head.metadata.nonEmpty)
    }

  }

  it should "ignore files that are not supported inside a directory" taggedAs SlowTest in {
    val supportedFiles = getSupportedFiles(mixDirectory)
    val reader2Image = new Reader2Image()
      .setContentPath(s"$mixDirectory")
      .setOutputCol("image")
      .setExplodeDocs(false)

    val pipeline = new Pipeline().setStages(Array(reader2Image))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualImageResult(resultDf, "image")

    assert(annotationsResult.length == supportedFiles.length)
    annotationsResult.foreach { annotations =>
      assert(annotations.head.annotatorType == AnnotatorType.IMAGE)
      assert(supportedFiles.contains(annotations.head.origin))
      assert(annotations.head.result.nonEmpty)
      assert(annotations.head.height > 0)
      assert(annotations.head.width > 0)
      assert(annotations.head.nChannels > 0)
      assert(annotations.head.mode > 0)
      assert(annotations.head.metadata.nonEmpty)
    }

  }

  it should "ignore unsupported files" taggedAs FastTest in {
    val reader2Image = new Reader2Image()
      .setContentPath(s"$unsupportedFiles")
      .setOutputCol("image")
      .setExplodeDocs(false)

    val pipeline = new Pipeline().setStages(Array(reader2Image))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() == 3)
    assert(resultDf.filter(col("exception").isNotNull).count() == 0)
  }

  it should "display error on exception column" taggedAs FastTest in {
    val reader2Image = new Reader2Image()
      .setContentPath(unsupportedFiles)
      .setOutputCol("image")
      .setIgnoreExceptions(false)

    val pipeline = new Pipeline().setStages(Array(reader2Image))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)
    assert(resultDf.filter(col("exception").isNotNull).count() > 1)
  }

  it should "output empty values when there is no image data" taggedAs FastTest in {
    val reader2Image = new Reader2Image()
      .setContentPath(s"$htmlFilesDirectory/example-div.html")
      .setContentType("text/html")
      .setOutputCol("image")

    val pipeline = new Pipeline().setStages(Array(reader2Image))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.isEmpty)
  }

  it should "work for email files with eml extension" taggedAs FastTest in {
    val emailFile = "email-test-image.eml"
    val emailPath = s"$emailDirectory/$emailFile"
    val reader2Image = new Reader2Image()
      .setContentPath(emailPath)
      .setContentType("message/rfc822")
      .setOutputCol("image")

    val pipeline = new Pipeline().setStages(Array(reader2Image))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualImageResult(resultDf, "image")
    assert(annotationsResult.length == 1)
    annotationsResult.foreach { annotations =>
      assert(annotations.head.annotatorType == AnnotatorType.IMAGE)
      assert(annotations.head.origin == emailFile)
      assert(annotations.head.origin == emailFile)
      assert(annotations.head.result.nonEmpty)
      assert(annotations.head.height > 0)
      assert(annotations.head.width > 0)
      assert(annotations.head.nChannels > 0)
      assert(annotations.head.mode > 0)
      assert(annotations.head.metadata.nonEmpty)
    }
  }

  it should "work for email files with msg extension" taggedAs FastTest in {
    val emailFile = "email-test-image.msg"
    val emailPath = s"$emailDirectory/$emailFile"
    val reader2Image = new Reader2Image()
      .setContentPath(emailPath)
      .setContentType("message/rfc822")
      .setOutputCol("image")

    val pipeline = new Pipeline().setStages(Array(reader2Image))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)
    val annotationsResult = AssertAnnotations.getActualImageResult(resultDf, "image")

    assert(annotationsResult.length == 1)
    annotationsResult.foreach { annotations =>
      assert(annotations.head.annotatorType == AnnotatorType.IMAGE)
      assert(annotations.head.origin == emailFile)
      assert(annotations.head.result.nonEmpty)
      assert(annotations.head.height > 0)
      assert(annotations.head.width > 0)
      assert(annotations.head.nChannels > 0)
      assert(annotations.head.mode > 0)
      assert(annotations.head.metadata.nonEmpty)
    }
  }

  it should "read images from MSWord files" taggedAs FastTest in {
    val wordFile = "contains-pictures.docx"
    val wordPath = s"$wordDirectory/$wordFile"
    val reader2Image = new Reader2Image()
      .setContentPath(wordPath)
      .setContentType("application/msword")
      .setOutputCol("image")

    val pipeline = new Pipeline().setStages(Array(reader2Image))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)
    val annotationsResult = AssertAnnotations.getActualImageResult(resultDf, "image")

    annotationsResult.foreach { annotations =>
      assert(annotations.head.annotatorType == AnnotatorType.IMAGE)
      assert(annotations.head.origin == wordFile)
      assert(annotations.head.result.nonEmpty)
      assert(annotations.head.height > 0)
      assert(annotations.head.width > 0)
      assert(annotations.head.nChannels > 0)
      assert(annotations.head.mode > 0)
      assert(annotations.head.metadata.nonEmpty)
    }
  }

  it should "read images from raw images files" taggedAs FastTest in {
    val imageFile = "SwitzerlandAlps.jpg"
    val imagePath = s"$imageDirectory/$imageFile"
    val reader2Image = new Reader2Image()
      .setContentPath(imagePath)
      .setContentType("image/raw")
      .setOutputCol("image")

    val pipeline = new Pipeline().setStages(Array(reader2Image))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    resultDf.show()
    val annotationsResult = AssertAnnotations.getActualImageResult(resultDf, "image")

    assert(annotationsResult.length == 1)
    annotationsResult.foreach { annotations =>
      assert(annotations.head.annotatorType == AnnotatorType.IMAGE)
      assert(annotations.head.origin == imageFile)
      assert(annotations.head.result.nonEmpty)
      assert(annotations.head.height > 0)
      assert(annotations.head.width > 0)
      assert(annotations.head.nChannels > 0)
      assert(annotations.head.mode > 0)
      assert(annotations.head.metadata.nonEmpty)
    }
  }

  it should "read a directory of mixed files and integrate with VLM models" taggedAs SlowTest in {
    // This pipeline requires 29GB of RAM to run
    val reader2Image = new Reader2Image()
      .setContentPath(filesDirectory)
      .setOutputCol("image")

    val pipeline = new Pipeline().setStages(Array(reader2Image))
    val pipelineModel = pipeline.fit(emptyDataSet)
    val imagesDf = pipelineModel.transform(emptyDataSet)
    imagesDf.show()

    val visualQAClassifier = Qwen2VLTransformer
      .pretrained()
      .setInputCols("image")
      .setOutputCol("answer")

    val vlmPipeline = new Pipeline().setStages(Array(visualQAClassifier))
    val resultDf = vlmPipeline.fit(imagesDf).transform(imagesDf)

    resultDf.select("image.origin", "answer.result").show(truncate = false)

    assert(!resultDf.isEmpty)
  }

  it should "add different user instructions to the prompt" taggedAs SlowTest in {
    val reader2Doc = new Reader2Image()
      .setContentPath(emailDirectory)
      .setOutputCol("image")
      .setUserMessage("Describe the image with 3 to 4 words.")

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val imagesDf = pipelineModel.transform(emptyDataSet)
    imagesDf.show()
    imagesDf.select("image.text").show(truncate = false)
    imagesDf.printSchema()

    val visualQAClassifier = Qwen2VLTransformer
      .pretrained()
      .setInputCols("image")
      .setOutputCol("answer")

    val vlmPipeline = new Pipeline().setStages(Array(visualQAClassifier))
    val resultDf = vlmPipeline.fit(imagesDf).transform(imagesDf)

    resultDf.select("image.origin", "answer.result").show(truncate = false)

    assert(!resultDf.isEmpty)
  }

  it should "work with SmolVLMTransformer" taggedAs SlowTest in {
    val reader2Doc = new Reader2Image()
      .setContentPath(emailDirectory)
      .setOutputCol("image")
      .setPromptTemplate("smolvl-chat")
      .setUserMessage("Are there cats in the image?")

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val imagesDf = pipelineModel.transform(emptyDataSet)
    imagesDf.show()
    imagesDf.select("image.text").show(truncate = false)
    imagesDf.printSchema()

    val visualQAClassifier = SmolVLMTransformer
      .pretrained()
      .setInputCols("image")
      .setOutputCol("answer")

    val vlmPipeline = new Pipeline().setStages(Array(visualQAClassifier))
    val resultDf = vlmPipeline.fit(imagesDf).transform(imagesDf)

    resultDf.select("image.origin", "answer.result").show(truncate = false)

    assert(!resultDf.isEmpty)
  }

  it should "infer for word files" taggedAs SlowTest in {
    val reader2Doc = new Reader2Image()
      .setContentPath(s"$wordDirectory/contains-pictures.docx")
      .setOutputCol("image")
      .setContentType("application/msword")
      .setUserMessage("Describe the image with 3 to 4 words.")

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val imagesDf = pipelineModel.transform(emptyDataSet)
    imagesDf.show()
    imagesDf.select("image.text").show(truncate = false)
    imagesDf.printSchema()

    val visualQAClassifier = Qwen2VLTransformer
      .pretrained()
      .setInputCols("image")
      .setOutputCol("answer")

    val vlmPipeline = new Pipeline().setStages(Array(visualQAClassifier))
    val resultDf = vlmPipeline.fit(imagesDf).transform(imagesDf)

    resultDf.select("image.origin", "answer.result").show(truncate = false)

    assert(!resultDf.isEmpty)
  }

  it should "infer for raw image files" taggedAs SlowTest in {
    val reader2Image = new Reader2Image()
      .setContentPath(s"$imageDirectory/SwitzerlandAlps.jpg")
      .setOutputCol("image")
      .setContentType("image/raw")

    val pipeline = new Pipeline().setStages(Array(reader2Image))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val imagesDf = pipelineModel.transform(emptyDataSet)
    imagesDf.show()
    imagesDf.select("image.text").show(truncate = false)
    imagesDf.printSchema()

    val visualQAClassifier = Qwen2VLTransformer
      .pretrained()
      .setInputCols("image")
      .setOutputCol("answer")

    val vlmPipeline = new Pipeline().setStages(Array(visualQAClassifier))
    val resultDf = vlmPipeline.fit(imagesDf).transform(imagesDf)

    resultDf.select("image.origin", "answer.result").show(truncate = false)

    assert(!resultDf.isEmpty)
  }

  it should "set custom prompt" taggedAs SlowTest in {
    val customPrompt = "<|im_start|><image>{prompt}<|im_end|><|im_start|>assistant"

    val reader2Doc = new Reader2Image()
      .setContentPath(emailDirectory)
      .setOutputCol("image")
      .setUserMessage("Describe the image with 3 to 4 words.")
      .setPromptTemplate("custom")
      .setCustomPromptTemplate(customPrompt)

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val imagesDf = pipelineModel.transform(emptyDataSet)
    imagesDf.show()
    imagesDf.select("image.text").show(truncate = false)
    imagesDf.printSchema()

    val visualQAClassifier = Qwen2VLTransformer
      .pretrained()
      .setInputCols("image")
      .setOutputCol("answer")

    val vlmPipeline = new Pipeline().setStages(Array(visualQAClassifier))
    val resultDf = vlmPipeline.fit(imagesDf).transform(imagesDf)

    resultDf.select("image.origin", "answer.result").show(truncate = false)

    assert(!resultDf.isEmpty)
  }

  it should "work with exception column" taggedAs SlowTest in {
    val reader2Doc = new Reader2Image()
      .setContentPath(unsupportedFiles)
      .setOutputCol("image")
      .setUserMessage("Describe the image with 3 to 4 words.")
      .setIgnoreExceptions(false)

    val pipeline = new Pipeline().setStages(Array(reader2Doc))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val imagesDf = pipelineModel.transform(emptyDataSet)

    imagesDf.show()
    imagesDf.select("image.text").show(truncate = false)
    imagesDf.printSchema()

    val visualQAClassifier = Qwen2VLTransformer
      .pretrained()
      .setInputCols("image")
      .setOutputCol("answer")

    val promptDf = imagesDf.filter(col("exception").isNull)
    val vlmPipeline = new Pipeline().setStages(Array(visualQAClassifier))
    val resultDf = vlmPipeline.fit(promptDf).transform(promptDf)

    resultDf.select("image.origin", "answer.result").show(truncate = false)
  }

  it should "add exception message to exception column" taggedAs FastTest in {
    val reader2Img = new Reader2Image()
      .setContentPath("src/test/resources/reader/pdf-corrupted/corrupted.pdf")
      .setContentType("application/pdf")
      .setOutputCol("image")
      .setIgnoreExceptions(false)
      .setUserMessage("Describe the image with 3 to 4 words.")

    val pipeline = new Pipeline().setStages(Array(reader2Img))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.filter(col("exception").isNotNull).count() == 1)
  }

  it should "output empty dataframe for unsupported files" taggedAs SlowTest in {
    val reader2Image = new Reader2Image()
      .setContentPath("src/test/resources/reader/csv")
      .setOutputCol("image")

    val pipeline = new Pipeline().setStages(Array(reader2Image))
    val pipelineModel = pipeline.fit(emptyDataSet)
    val imagesDf = pipelineModel.transform(emptyDataSet)

    assert(imagesDf.isEmpty)
  }

  it should "extract images from PDF" taggedAs FastTest in {
    val sourceFile = "pdf-with-2images.pdf"
    val reader2Image = new Reader2Image()
      .setContentPath(s"$pdfDirectory/$sourceFile")
      .setContentType("application/pdf")
      .setOutputCol("image")

    val pipeline = new Pipeline().setStages(Array(reader2Image))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)
    resultDf.show()
    val annotationsResult = AssertAnnotations.getActualImageResult(resultDf, "image")

    assert(annotationsResult.length == 2)
    annotationsResult.foreach { annotations =>
      assert(annotations.head.annotatorType == AnnotatorType.IMAGE)
      assert(annotations.head.origin == sourceFile)
      assert(annotations.head.result.nonEmpty)
      assert(annotations.head.height > 0)
      assert(annotations.head.width > 0)
      assert(annotations.head.nChannels > 0)
      assert(annotations.head.mode > 0)
      assert(annotations.head.metadata.nonEmpty)
    }
  }

  it should "integrate PDF images output with VLM models" taggedAs SlowTest in {
    val sourceFile = "pdf-with-2images.pdf"
    val reader2Image = new Reader2Image()
      .setContentPath(s"$pdfDirectory/$sourceFile")
      .setContentType("application/pdf")
      .setOutputCol("image")

    val pipeline = new Pipeline().setStages(Array(reader2Image))
    val pipelineModel = pipeline.fit(emptyDataSet)
    val imagesDf = pipelineModel.transform(emptyDataSet)
    imagesDf.show()

    val visualQAClassifier = Qwen2VLTransformer
      .pretrained()
      .setInputCols("image")
      .setOutputCol("answer")

    val vlmPipeline = new Pipeline().setStages(Array(visualQAClassifier))
    val resultDf = vlmPipeline.fit(imagesDf).transform(imagesDf)

    resultDf.select("image.origin", "answer.result").show(truncate = false)

    assert(!resultDf.isEmpty)
  }

  def getSupportedFiles(dirPath: String): Seq[String] = {
    val supportedExtensions = Seq(".html", ".htm", ".md", "doc", "docx")

    val dir = new File(dirPath)
    if (dir.exists && dir.isDirectory) {
      dir.listFiles
        .filter(f =>
          f.isFile && supportedExtensions.exists(ext => f.getName.toLowerCase.endsWith(ext)))
        .toSeq
        .map(_.getName)
    } else {
      Seq.empty
    }
  }

}
