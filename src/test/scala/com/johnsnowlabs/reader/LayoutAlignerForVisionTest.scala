package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFVisionModel
import com.johnsnowlabs.nlp.{AnnotatorType, AssertAnnotations}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class LayoutAlignerForVisionTest extends AnyFlatSpec with SparkSessionTest {

  val docDirectory = "src/test/resources/reader/doc"
  val pptDirectory = "src/test/resources/reader/ppt"
  val pdfDirectory = "src/test/resources/reader/pdf"
  val baseImagePrompt =
    "Describe in a short and easy to understand sentence what you see in the image"

  "LayoutAlignerForVision" should "emit paired doc and image columns by default" taggedAs FastTest in {

    val reader = new ReaderAssembler()
      .setContentType("application/msword")
      .setContentPath(s"$docDirectory/contains-pictures.docx")
      .setOutputAsDocument(false)
      .setOutputCol("data")

    val aligner = new LayoutAlignerForVision()
      .setInputCols("data_text", "data_image")
      .setOutputCol("aligned")

    val pipeline = new Pipeline().setStages(Array(reader, aligner))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    assert(resultDf.columns.contains("aligned_doc"))
    assert(resultDf.columns.contains("aligned_image"))
    assert(resultDf.columns.contains("aligned_prompt"))
    assert(
      resultDf
        .schema("aligned_doc")
        .metadata
        .getString("annotatorType") == AnnotatorType.DOCUMENT)
    assert(
      resultDf.schema("aligned_image").metadata.getString("annotatorType") == AnnotatorType.IMAGE)
    assert(
      resultDf
        .schema("aligned_prompt")
        .metadata
        .getString("annotatorType") == AnnotatorType.DOCUMENT)

    val alignedDocs = AssertAnnotations.getActualResult(resultDf, "aligned_doc")
    val alignedImages = AssertAnnotations.getActualImageResult(resultDf, "aligned_image")
    val alignedPrompts = AssertAnnotations.getActualResult(resultDf, "aligned_prompt")

    assert(alignedDocs.nonEmpty)
    assert(alignedDocs.length == alignedImages.length)
    assert(alignedDocs.length == alignedPrompts.length)
    assert(
      alignedDocs.forall(_.size == 1),
      "Each exploded row should have a single doc annotation")
    assert(
      alignedImages.forall(_.size == 1),
      "Each exploded row should have a single image annotation")
    assert(
      alignedPrompts.forall(_.size == 1),
      "Each exploded row should have a single prompt annotation")
    assert(
      alignedDocs.forall(!_.head.metadata.contains("distance")),
      "Doc metadata should not duplicate alignment distance")
    assert(
      alignedImages.forall(_.head.metadata.contains("distance")),
      "Image metadata should keep alignment distance")
    assert(
      alignedImages.forall(_.head.metadata.contains("match_strategy")),
      "Image metadata should keep match strategy")
    assert(
      alignedPrompts.forall(_.head.result == baseImagePrompt),
      "Default prompt should describe only the image")
  }

  it should "keep a single primary image and all image_matches when mergeImagesPerChunk is enabled" taggedAs FastTest in {

    val reader = new ReaderAssembler()
      .setContentType("application/msword")
      .setContentPath(s"$docDirectory/contains-pictures.docx")
      .setOutputAsDocument(false)
      .setOutputCol("data")

    val aligner = new LayoutAlignerForVision()
      .setInputCols("data_text", "data_image")
      .setOutputCol("aligned")
      .setMergeImagesPerChunk(true)
      .setExplodeDocs(true)

    val pipeline = new Pipeline().setStages(Array(reader, aligner))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    val alignedDocs = AssertAnnotations.getActualResult(resultDf, "aligned_doc")
    val alignedImages = AssertAnnotations.getActualImageResult(resultDf, "aligned_image")

    assert(alignedDocs.length == 4, "Expected one merged doc/image pair per paragraph")
    assert(alignedDocs.length == alignedImages.length)
    assert(alignedDocs.forall(_.size == 1))
    assert(alignedImages.forall(_.size == 1))
    assert(
      alignedDocs.forall(_.head.metadata.contains("image_matches")),
      "Merged doc metadata should include image_matches")
  }

  it should "include aligned text in prompt when addNeighborText is enabled" taggedAs FastTest in {

    val reader = new ReaderAssembler()
      .setContentType("application/msword")
      .setContentPath(s"$docDirectory/contains-pictures.docx")
      .setOutputAsDocument(false)
      .setOutputCol("data")

    val aligner = new LayoutAlignerForVision()
      .setInputCols("data_text", "data_image")
      .setOutputCol("aligned")
      .setAddNeighborText(true)

    val pipeline = new Pipeline().setStages(Array(reader, aligner))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)
    resultDf.select("aligned_doc", "aligned_image.metadata").show(truncate = false)

    val alignedDocs = AssertAnnotations.getActualResult(resultDf, "aligned_doc")
    val alignedPrompts = AssertAnnotations.getActualResult(resultDf, "aligned_prompt")

    assert(alignedDocs.length == alignedPrompts.length)
    assert(
      alignedDocs.zip(alignedPrompts).forall { case (docs, prompts) =>
        val docText = docs.head.result
        val promptText = prompts.head.result
        promptText.contains("and then summarize it along with this text:") &&
        promptText.contains(docText)
      },
      "Prompt should include aligned text when addNeighborText=true")
  }

  it should "align PowerPoint text/images on the same slide" taggedAs FastTest in {

    val reader = new ReaderAssembler()
      .setContentType("application/vnd.ms-powerpoint")
      .setContentPath(s"$pptDirectory/power-point-images.pptx")
      .setOutputAsDocument(false)
      .setOutputCol("data")

    val aligner = new LayoutAlignerForVision()
      .setInputCols("data_text", "data_image")
      .setOutputCol("aligned")

    val pipeline = new Pipeline().setStages(Array(reader, aligner))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    assert(resultDf.columns.contains("aligned_doc"))
    assert(resultDf.columns.contains("aligned_image"))
    assert(resultDf.columns.contains("aligned_prompt"))

    val alignedDocs = AssertAnnotations.getActualResult(resultDf, "aligned_doc")
    val alignedImages = AssertAnnotations.getActualImageResult(resultDf, "aligned_image")
    val alignedPrompts = AssertAnnotations.getActualResult(resultDf, "aligned_prompt")

    assert(alignedDocs.nonEmpty, "Expected aligned document/image pairs for PowerPoint")
    assert(
      alignedDocs.length == alignedImages.length,
      "Expected aligned_doc and aligned_image row counts to match")
    assert(
      alignedDocs.length == alignedPrompts.length,
      "Expected aligned_doc and aligned_prompt row counts to match")
    assert(alignedDocs.forall(_.size == 1), "Each exploded row should have one doc annotation")
    assert(
      alignedImages.forall(_.size == 1),
      "Each exploded row should have one image annotation")
    assert(
      alignedDocs.forall(_.head.metadata.contains("slide_index")),
      "Aligned document metadata should include slide_index")
    assert(
      alignedImages.forall(_.head.metadata.contains("slide_index")),
      "Aligned image metadata should include slide_index")
    assert(
      alignedImages.forall(_.head.metadata.contains("match_strategy")),
      "Aligned image metadata should include match_strategy")
    assert(
      alignedPrompts.forall(_.head.result == baseImagePrompt),
      "Default prompt should describe only the image")
    assert(
      alignedDocs
        .zip(alignedImages)
        .forall { case (docs, images) =>
          docs.head.metadata.get("slide_index") == images.head.metadata.get("slide_index")
        },
      "Aligned text/image should belong to the same slide")
  }

  it should "align PDF text/images on the same page when image coordinates are page-scale" taggedAs FastTest in {
    val reader = new ReaderAssembler()
      .setContentType("application/pdf")
      .setContentPath(s"$pdfDirectory/pdf-with-2images.pdf")
      .setOutputAsDocument(false)
      .setOutputCol("data")

    val aligner = new LayoutAlignerForVision()
      .setInputCols("data_text", "data_image")
      .setOutputCol("aligned")

    val pipeline = new Pipeline().setStages(Array(reader, aligner))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    val alignedDocs = AssertAnnotations.getActualResult(resultDf, "aligned_doc")
    val alignedImages = AssertAnnotations.getActualImageResult(resultDf, "aligned_image")

    assert(alignedDocs.nonEmpty, "Expected aligned document/image pairs for PDF")
    assert(
      alignedDocs.length == alignedImages.length,
      "Expected aligned_doc and aligned_image row counts to match")
    assert(alignedDocs.forall(_.size == 1), "Each exploded row should have one doc annotation")
    assert(
      alignedImages.forall(_.size == 1),
      "Each exploded row should have one image annotation")
    assert(
      alignedDocs.forall(_.head.metadata.contains("pageNumber")),
      "Aligned document metadata should include pageNumber")
    assert(
      alignedImages.forall(_.head.metadata.contains("pageNumber")),
      "Aligned image metadata should include pageNumber")
    assert(
      alignedImages.forall(
        _.head.metadata.get("match_strategy").exists(_.startsWith("same_page"))),
      "PDF alignments should use same_page strategies")
    assert(
      alignedDocs
        .zip(alignedImages)
        .forall { case (docs, images) =>
          docs.head.metadata.get("pageNumber") == images.head.metadata.get("pageNumber")
        },
      "Aligned text/image should belong to the same PDF page")
  }

  it should "work with AutoGGUFVisionModel for doc files" taggedAs SlowTest in {

    val reader = new ReaderAssembler()
      .setContentType("application/msword")
      .setContentPath(s"$docDirectory/contains-pictures.docx")
      .setOutputAsDocument(false)
      .setOutputCol("data")
      .setUseEncodedImageBytes(true)

    val aligner = new LayoutAlignerForVision()
      .setInputCols("data_text", "data_image")
      .setOutputCol("aligned")
      .setExplodeDocs(true)

    val autoGgufModel: AutoGGUFVisionModel = AutoGGUFVisionModel
      .pretrained()
      .setInputCols("aligned_prompt", "aligned_image")
      .setOutputCol("caption")
      .setBatchSize(2)
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

    val pipeline = new Pipeline().setStages(Array(reader, aligner, autoGgufModel))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    resultDf.select("aligned_prompt").show(truncate = false)
    resultDf.select("aligned_doc", "caption").show(truncate = false)
  }

}
