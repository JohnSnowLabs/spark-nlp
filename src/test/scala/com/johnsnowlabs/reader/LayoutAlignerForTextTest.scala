/*
 * Copyright 2017-2026 John Snow Labs
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
import com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFVisionModel
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, AssertAnnotations}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.MetadataBuilder
import org.scalatest.flatspec.AnyFlatSpec

class LayoutAlignerForTextTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  private val docMetadata =
    new MetadataBuilder().putString("annotatorType", AnnotatorType.DOCUMENT).build()

  private def docAnnotation(
      begin: Int,
      end: Int,
      text: String,
      metadata: Map[String, String]): Annotation =
    Annotation(
      annotatorType = AnnotatorType.DOCUMENT,
      begin = begin,
      end = end,
      result = text,
      metadata = metadata,
      embeddings = Array.emptyFloatArray)

  private def buildAutoGgufVisionModel(batchSize: Int): AutoGGUFVisionModel =
    AutoGGUFVisionModel
      .pretrained()
      .setInputCols("aligned_prompt", "aligned_image")
      .setOutputCol("image_caption")
      .setBatchSize(batchSize)
      .setNGpuLayers(99)
      .setNCtx(4096)
      .setMinKeep(0)
      .setMinP(0.05f)
      .setNPredict(40)
      .setPenalizeNl(true)
      .setRepeatPenalty(1.18f)
      .setTemperature(0.05f)
      .setTopK(40)
      .setTopP(0.95f)

  "LayoutAlignerForText" should "rebuild text grouped by element_id and keep consistent indexes" taggedAs FastTest in {

    val docText1 = "Some paragraph text that runs through the floating picture."
    val docText2 = "An inline picture appears here and this text follows it on the same line."
    val inlineCaption1 = "The Python logo appears prominently on this page."
    val floatingCaption1 = "A Monty Python poster appears on the right side."
    val inlineCaption2 = "Another Python-powered inline logo is shown."

    val input = Seq(
      (
        "contains-pictures.docx",
        Seq(
          docAnnotation(
            32,
            32 + docText1.length - 1,
            docText1,
            Map(
              "element_id" -> "e1",
              "paragraph_index" -> "1",
              "paragraph_y" -> "25",
              "elementType" -> "NarrativeText"))),
        Seq(
          docAnnotation(
            0,
            floatingCaption1.length - 1,
            floatingCaption1,
            Map(
              "orderImageIndex" -> "2",
              "distance" -> "19",
              "confidence" -> "0.75",
              "image_type" -> "floating",
              "coord" -> "{x:212,y:6}",
              "source_file" -> "contains-pictures.docx")))),
      (
        "contains-pictures.docx",
        Seq(
          docAnnotation(
            32,
            32 + docText1.length - 1,
            docText1,
            Map(
              "element_id" -> "e1",
              "paragraph_index" -> "1",
              "paragraph_y" -> "25",
              "elementType" -> "NarrativeText"))),
        Seq(
          docAnnotation(
            0,
            inlineCaption1.length - 1,
            inlineCaption1,
            Map(
              "orderImageIndex" -> "1",
              "distance" -> "25",
              "confidence" -> "0.75",
              "image_type" -> "inline",
              "coord" -> "{x:0,y:0}",
              "source_file" -> "contains-pictures.docx")))),
      (
        "contains-pictures.docx",
        Seq(
          docAnnotation(
            430,
            430 + docText2.length - 1,
            docText2,
            Map(
              "element_id" -> "e2",
              "paragraph_index" -> "3",
              "paragraph_y" -> "75",
              "elementType" -> "NarrativeText"))),
        Seq(
          docAnnotation(
            0,
            inlineCaption2.length - 1,
            inlineCaption2,
            Map(
              "orderImageIndex" -> "3",
              "distance" -> "5",
              "confidence" -> "0.95",
              "image_type" -> "inline",
              "coord" -> "{x:0,y:80}",
              "source_file" -> "contains-pictures.docx")))))
      .toDF("fileName", "aligned_doc", "image_caption")
      .withColumn("aligned_doc", col("aligned_doc").as("aligned_doc", docMetadata))
      .withColumn("image_caption", col("image_caption").as("image_caption", docMetadata))

    val aligner = new LayoutAlignerForText()
      .setInputCols("aligned_doc", "image_caption")
      .setOutputCol("aligned_text")
      .setExplodeElements(true)

    val resultDf = aligner.transform(input)

    assert(resultDf.columns.contains("aligned_text"))
    assert(resultDf.count() == 2)
    assert(
      resultDf
        .schema("aligned_text")
        .metadata
        .getString("annotatorType") == AnnotatorType.DOCUMENT)

    val aligned = AssertAnnotations.getActualResult(resultDf, "aligned_text")
    assert(aligned.length == 2)
    assert(aligned.forall(_.size == 1))

    val rebuiltByElement = aligned.map(_.head).map(a => a.metadata("element_id") -> a).toMap
    assert(rebuiltByElement.keySet == Set("e1", "e2"))

    val rebuiltE1 = rebuiltByElement("e1")
    val expectedE1 = Seq(inlineCaption1, docText1, floatingCaption1).mkString("\n")
    assert(rebuiltE1.result == expectedE1)
    assert(rebuiltE1.begin == 0)
    assert(rebuiltE1.end == rebuiltE1.result.length - 1)
    assert(rebuiltE1.metadata.get("layout_aligner").contains("LayoutAlignerForText"))

    val rebuiltE2 = rebuiltByElement("e2")
    val expectedE2 = Seq(inlineCaption2, docText2).mkString("\n")
    assert(rebuiltE2.result == expectedE2)
    assert(rebuiltE2.begin == 0)
    assert(rebuiltE2.end == rebuiltE2.result.length - 1)
  }

  it should "raise an exception when fileName is missing" taggedAs FastTest in {
    val input = Seq(
      (
        Seq(
          docAnnotation(0, 10, "doc text", Map("element_id" -> "e1", "paragraph_index" -> "1"))),
        Seq(docAnnotation(
          0,
          12,
          "caption text",
          Map(
            "orderImageIndex" -> "1",
            "distance" -> "1",
            "confidence" -> "0.95",
            "image_type" -> "inline",
            "coord" -> "{x:0,y:10}",
            "source_file" -> "contains-pictures.docx")))))
      .toDF("aligned_doc", "image_caption")
      .withColumn("aligned_doc", col("aligned_doc").as("aligned_doc", docMetadata))
      .withColumn("image_caption", col("image_caption").as("image_caption", docMetadata))

    val aligner = new LayoutAlignerForText()
      .setInputCols("aligned_doc", "image_caption")
      .setOutputCol("aligned_text")
      .setExplodeElements(true)

    val exception = intercept[IllegalArgumentException] {
      aligner.transform(input)
    }

    assert(exception.getMessage.contains("fileName"))
  }

  it should "drop input annotation columns by default and use aligned_text as authoritative output" taggedAs FastTest in {

    val input = Seq(
      (
        "contains-pictures.docx",
        Seq(
          docAnnotation(
            0,
            10,
            "first text",
            Map("element_id" -> "e1", "paragraph_index" -> "1", "paragraph_y" -> "10"))),
        Seq(
          docAnnotation(
            0,
            11,
            "first caption",
            Map(
              "orderImageIndex" -> "1",
              "distance" -> "1",
              "confidence" -> "0.95",
              "image_type" -> "inline",
              "coord" -> "{x:0,y:10}",
              "source_file" -> "contains-pictures.docx")))),
      (
        "contains-pictures.docx",
        Seq(
          docAnnotation(
            20,
            30,
            "second text",
            Map("element_id" -> "e2", "paragraph_index" -> "2", "paragraph_y" -> "20"))),
        Seq(
          docAnnotation(
            0,
            12,
            "second caption",
            Map(
              "orderImageIndex" -> "2",
              "distance" -> "2",
              "confidence" -> "0.95",
              "image_type" -> "floating",
              "coord" -> "{x:120,y:20}",
              "source_file" -> "contains-pictures.docx")))))
      .toDF("fileName", "aligned_doc", "image_caption")
      .withColumn("aligned_doc", col("aligned_doc").as("aligned_doc", docMetadata))
      .withColumn("image_caption", col("image_caption").as("image_caption", docMetadata))

    val aligner = new LayoutAlignerForText()
      .setInputCols("aligned_doc", "image_caption")
      .setOutputCol("aligned_text")
      .setExplodeElements(true)

    val resultDf = aligner.transform(input)

    assert(resultDf.columns.contains("aligned_text"))
    assert(!resultDf.columns.contains("aligned_doc"))
    assert(!resultDf.columns.contains("image_caption"))
    assert(resultDf.count() == 2)
  }

  it should "preserve grouped input annotation columns when preserveColumns is enabled" taggedAs FastTest in {

    val input = Seq(
      (
        "contains-pictures.docx",
        Seq(
          docAnnotation(
            0,
            10,
            "first text",
            Map("element_id" -> "e1", "paragraph_index" -> "1", "paragraph_y" -> "10"))),
        Seq(
          docAnnotation(
            0,
            11,
            "first caption",
            Map(
              "orderImageIndex" -> "1",
              "distance" -> "1",
              "confidence" -> "0.95",
              "image_type" -> "inline",
              "coord" -> "{x:0,y:10}",
              "source_file" -> "contains-pictures.docx")))),
      (
        "contains-pictures.docx",
        Seq(
          docAnnotation(
            20,
            30,
            "second text",
            Map("element_id" -> "e2", "paragraph_index" -> "2", "paragraph_y" -> "20"))),
        Seq(
          docAnnotation(
            0,
            12,
            "second caption",
            Map(
              "orderImageIndex" -> "2",
              "distance" -> "2",
              "confidence" -> "0.95",
              "image_type" -> "floating",
              "coord" -> "{x:120,y:20}",
              "source_file" -> "contains-pictures.docx")))))
      .toDF("fileName", "aligned_doc", "image_caption")
      .withColumn("aligned_doc", col("aligned_doc").as("aligned_doc", docMetadata))
      .withColumn("image_caption", col("image_caption").as("image_caption", docMetadata))

    val aligner = new LayoutAlignerForText()
      .setInputCols("aligned_doc", "image_caption")
      .setOutputCol("aligned_text")
      .setExplodeElements(true)
      .setPreserveColumns(true)

    val resultDf = aligner.transform(input)

    assert(resultDf.columns.contains("aligned_text"))
    assert(resultDf.columns.contains("aligned_doc"))
    assert(resultDf.columns.contains("image_caption"))
    assert(resultDf.count() == 2)

    val preservedDocs = AssertAnnotations.getActualResult(resultDf, "aligned_doc")
    val preservedCaptions = AssertAnnotations.getActualResult(resultDf, "image_caption")
    assert(preservedDocs.forall(_.size == 2))
    assert(preservedCaptions.forall(_.size == 2))
  }

  it should "deduplicate repeated image matches by keeping the closest assignment" taggedAs FastTest in {

    val docText1 = "Text before the best image match."
    val docText2 = "Text with the closest aligned image."
    val repeatedCaption = "Python powered logo."

    val input = Seq(
      (
        "contains-pictures.docx",
        Seq(
          docAnnotation(
            100,
            100 + docText1.length - 1,
            docText1,
            Map("element_id" -> "e10", "paragraph_index" -> "4", "paragraph_y" -> "100"))),
        Seq(
          docAnnotation(
            0,
            repeatedCaption.length - 1,
            repeatedCaption,
            Map(
              "orderImageIndex" -> "5",
              "distance" -> "25",
              "confidence" -> "0.75",
              "image_type" -> "floating",
              "coord" -> "{x:25,y:125}",
              "source_file" -> "contains-pictures.docx")))),
      (
        "contains-pictures.docx",
        Seq(
          docAnnotation(
            200,
            200 + docText2.length - 1,
            docText2,
            Map("element_id" -> "e11", "paragraph_index" -> "5", "paragraph_y" -> "125"))),
        Seq(
          docAnnotation(
            0,
            repeatedCaption.length - 1,
            repeatedCaption,
            Map(
              "orderImageIndex" -> "5",
              "distance" -> "0",
              "confidence" -> "0.95",
              "image_type" -> "floating",
              "coord" -> "{x:25,y:125}",
              "source_file" -> "contains-pictures.docx")))))
      .toDF("fileName", "aligned_doc", "image_caption")
      .withColumn("aligned_doc", col("aligned_doc").as("aligned_doc", docMetadata))
      .withColumn("image_caption", col("image_caption").as("image_caption", docMetadata))

    val aligner = new LayoutAlignerForText()
      .setInputCols("aligned_doc", "image_caption")
      .setOutputCol("aligned_text")
      .setExplodeElements(true)

    val resultDf = aligner.transform(input)
    val aligned = AssertAnnotations.getActualResult(resultDf, "aligned_text")
    assert(aligned.length == 2)
    assert(aligned.forall(_.size == 1))
    val rebuiltByElement = aligned.map(_.head).map(a => a.metadata("element_id") -> a).toMap
    assert(rebuiltByElement.keySet == Set("e10", "e11"))

    val rebuiltE10 = rebuiltByElement("e10")
    val rebuiltE11 = rebuiltByElement("e11")

    assert(rebuiltE10.result == docText1)
    assert(rebuiltE11.result == Seq(docText2, repeatedCaption).mkString("\n"))

    val allText = Seq(rebuiltE10.result, rebuiltE11.result).mkString("\n")
    assert(allText.split(repeatedCaption, -1).length - 1 == 1)
  }

  it should "keep all text chunks when captions are fewer than docs" taggedAs FastTest in {

    val heading = "Revenue Report"
    val paragraph1 = "North America showed stable growth."
    val paragraph2 = "Europe and APAC both improved quarter over quarter."
    val caption = "Bar chart compares quarterly regional revenue."

    val input = Seq(
      (
        "report.pdf",
        Seq(
          docAnnotation(
            0,
            heading.length - 1,
            heading,
            Map("element_id" -> "e0", "paragraph_index" -> "0", "paragraph_y" -> "0")),
          docAnnotation(
            20,
            20 + paragraph1.length - 1,
            paragraph1,
            Map("element_id" -> "e1", "paragraph_index" -> "1", "paragraph_y" -> "25")),
          docAnnotation(
            80,
            80 + paragraph2.length - 1,
            paragraph2,
            Map("element_id" -> "e2", "paragraph_index" -> "2", "paragraph_y" -> "50"))),
        Seq(docAnnotation(
          0,
          caption.length - 1,
          caption,
          Map(
            "paragraph_index" -> "1",
            "orderImageIndex" -> "1",
            "distance" -> "0",
            "confidence" -> "0.95",
            "image_type" -> "floating",
            "coord" -> "{x:120,y:25}",
            "source_file" -> "report.pdf")))))
      .toDF("fileName", "aligned_doc", "image_caption")
      .withColumn("aligned_doc", col("aligned_doc").as("aligned_doc", docMetadata))
      .withColumn("image_caption", col("image_caption").as("image_caption", docMetadata))

    val aligner = new LayoutAlignerForText()
      .setInputCols("aligned_doc", "image_caption")
      .setOutputCol("aligned_text")
      .setExplodeElements(true)

    val resultDf = aligner.transform(input)
    val aligned = AssertAnnotations.getActualResult(resultDf, "aligned_text").flatten

    assert(aligned.size == 3)

    val rebuiltByElement = aligned.map(a => a.metadata("element_id") -> a).toMap
    assert(rebuiltByElement("e0").result == heading)
    assert(rebuiltByElement("e1").result == Seq(paragraph1, caption).mkString("\n"))
    assert(rebuiltByElement("e2").result == paragraph2)
  }

  it should "return a single file-level annotation by default" taggedAs FastTest in {
    val heading = "Revenue Report"
    val paragraph1 = "North America showed stable growth."
    val paragraph2 = "Europe and APAC both improved quarter over quarter."
    val caption = "Bar chart compares quarterly regional revenue."

    val input = Seq(
      (
        "report.pdf",
        Seq(
          docAnnotation(
            0,
            heading.length - 1,
            heading,
            Map("element_id" -> "e0", "paragraph_index" -> "0", "paragraph_y" -> "0")),
          docAnnotation(
            20,
            20 + paragraph1.length - 1,
            paragraph1,
            Map("element_id" -> "e1", "paragraph_index" -> "1", "paragraph_y" -> "25")),
          docAnnotation(
            80,
            80 + paragraph2.length - 1,
            paragraph2,
            Map("element_id" -> "e2", "paragraph_index" -> "2", "paragraph_y" -> "50"))),
        Seq(docAnnotation(
          0,
          caption.length - 1,
          caption,
          Map(
            "paragraph_index" -> "1",
            "orderImageIndex" -> "1",
            "distance" -> "0",
            "confidence" -> "0.95",
            "image_type" -> "floating",
            "coord" -> "{x:120,y:25}",
            "source_file" -> "report.pdf")))))
      .toDF("fileName", "aligned_doc", "image_caption")
      .withColumn("aligned_doc", col("aligned_doc").as("aligned_doc", docMetadata))
      .withColumn("image_caption", col("image_caption").as("image_caption", docMetadata))

    val aligner = new LayoutAlignerForText()
      .setInputCols("aligned_doc", "image_caption")
      .setOutputCol("aligned_text")

    val resultDf = aligner.transform(input)
    assert(resultDf.count() == 1)

    val aligned = AssertAnnotations.getActualResult(resultDf, "aligned_text")
    assert(aligned.size == 1)
    assert(aligned.head.size == 1)

    val mergedText = aligned.head.head.result
    assert(mergedText.contains(heading))
    assert(mergedText.contains(paragraph1))
    assert(mergedText.contains(caption))
    assert(mergedText.contains(paragraph2))
    assert(mergedText.indexOf(paragraph1) < mergedText.indexOf(caption))
  }

  it should "work with AutoGGUFVisionModel for doc files" taggedAs SlowTest in {
    val docDirectory = "src/test/resources/reader/doc"

    val reader = new ReaderAssembler()
      .setContentType("application/msword")
      .setContentPath(s"$docDirectory/contains-pictures.docx")
      .setOutputAsDocument(false)
      .setOutputCol("data")
      .setUseEncodedImageBytes(true)

    val alignerVision = new LayoutAlignerForVision()
      .setInputCols("data_text", "data_image")
      .setOutputCol("aligned")

    val autoGgufModel = buildAutoGgufVisionModel(batchSize = 2)

    // try to preserve columns but is not guaranteed to work, since it groups the columns are changed
    val alignerText = new LayoutAlignerForText()
      .setInputCols("aligned_doc", "image_caption")
      .setOutputCol("aligned_text")
      .setPreserveColumns(true)

    val pipeline =
      new Pipeline().setStages(Array(reader, alignerVision, autoGgufModel, alignerText))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet).cache()
    resultDf.count()

    try {
      resultDf.select("data_text").show(truncate = false)
      resultDf.select("aligned_prompt").show(truncate = false)
      resultDf.select("aligned_doc", "image_caption.result").show(truncate = false)
      resultDf.select("aligned_text").show(truncate = false)

      resultDf
        .selectExpr(
          "size(aligned_doc) as n_docs",
          "size(image_caption) as n_caps",
          "size(aligned_text) as n_text")
        .show(false)
    } finally {
      resultDf.unpersist(blocking = false)
    }

  }

  it should "work with AutoGGUFVisionModel for doc files and drop columns" taggedAs SlowTest in {
    val docDirectory = "src/test/resources/reader/doc"

    val reader = new ReaderAssembler()
      .setContentType("application/msword")
      .setContentPath(s"$docDirectory/contains-pictures.docx")
      .setOutputAsDocument(false)
      .setOutputCol("data")
      .setUseEncodedImageBytes(true)

    val alignerVision = new LayoutAlignerForVision()
      .setInputCols("data_text", "data_image")
      .setOutputCol("aligned")

    val autoGgufModel = buildAutoGgufVisionModel(batchSize = 2)

    // By default, it drops  image_caption (output from AutoGGUFVisionModel) and aligned_doc(output from alignerVision) and only keeps the aligned_text as output
    val alignerText = new LayoutAlignerForText()
      .setInputCols("aligned_doc", "image_caption")
      .setOutputCol("aligned_text")

    val pipeline =
      new Pipeline().setStages(Array(reader, alignerVision, autoGgufModel, alignerText))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet).cache()
    resultDf.count()

    try {
      resultDf.select("data_text").show(truncate = false)
      resultDf.select("aligned_prompt").show(truncate = false)
      resultDf.select("aligned_text").show(truncate = false)
    } finally {
      resultDf.unpersist(blocking = false)
    }
  }

  it should "work with AutoGGUFVisionModel in a separate pipeline with PDF files" taggedAs SlowTest in {
    val pdfDirectory = "src/test/resources/reader/pdf"

    val reader = new ReaderAssembler()
      .setContentType("application/pdf")
      .setContentPath(s"$pdfDirectory/pdf-with-2images.pdf")
      .setOutputAsDocument(false)
      .setOutputCol("data")
      .setUseEncodedImageBytes(true)

    val alignerVision = new LayoutAlignerForVision()
      .setInputCols("data_text", "data_image")
      .setOutputCol("aligned")

    val autoGgufModel = buildAutoGgufVisionModel(batchSize = 1)

    val imageCaptionPipeline =
      new Pipeline().setStages(Array(reader, alignerVision, autoGgufModel))
    val imageCaptionDf = imageCaptionPipeline.fit(emptyDataSet).transform(emptyDataSet).cache()
    imageCaptionDf.count()

    try {
      imageCaptionDf
        .select("aligned_prompt.result", "data_text", "aligned_doc", "image_caption")
        .show(truncate = false)

      val alignerText = new LayoutAlignerForText()
        .setInputCols("aligned_doc", "image_caption")
        .setOutputCol("aligned_text")

      val pipeline =
        new Pipeline().setStages(Array(alignerText))
      val resultDf = pipeline.fit(imageCaptionDf).transform(imageCaptionDf).cache()
      resultDf.count()

      try {
        resultDf.select("aligned_text").show(truncate = false)
      } finally {
        resultDf.unpersist(blocking = false)
      }
    } finally {
      imageCaptionDf.unpersist(blocking = false)
    }
  }

}
