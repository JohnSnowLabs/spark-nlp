package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFVisionModel
import com.johnsnowlabs.nlp.{AnnotatorType, AssertAnnotations}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class LayoutAlignerForVisionTest extends AnyFlatSpec with SparkSessionTest {

  val docDirectory = "src/test/resources/reader/doc"

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
    resultDf.show()
    resultDf.select("aligned_doc", "aligned_image.metadata").show(truncate = false)
    resultDf.printSchema()

    assert(resultDf.columns.contains("aligned_doc"))
    assert(resultDf.columns.contains("aligned_image"))
    assert(
      resultDf
        .schema("aligned_doc")
        .metadata
        .getString("annotatorType") == AnnotatorType.DOCUMENT)
    assert(
      resultDf.schema("aligned_image").metadata.getString("annotatorType") == AnnotatorType.IMAGE)

    val alignedDocs = AssertAnnotations.getActualResult(resultDf, "aligned_doc")
    val alignedImages = AssertAnnotations.getActualImageResult(resultDf, "aligned_image")

    assert(alignedDocs.nonEmpty)
    assert(alignedDocs.length == alignedImages.length)
    assert(
      alignedDocs.forall(_.size == 1),
      "Each exploded row should have a single doc annotation")
    assert(
      alignedImages.forall(_.size == 1),
      "Each exploded row should have a single image annotation")
    assert(
      alignedDocs.forall(_.head.metadata.contains("distance")),
      "Doc metadata should keep alignment distance")
    assert(
      alignedImages.forall(_.head.metadata.contains("distance")),
      "Image metadata should keep alignment distance")
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

  it should "work with AutoGGUFVisionModel" taggedAs FastTest in {

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
      .setInputCols("data_text", "data_image")
      .setOutputCol("completions")
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

    resultDf.select("aligned_doc.result", "completions.result").show(truncate = false)

  }

}
