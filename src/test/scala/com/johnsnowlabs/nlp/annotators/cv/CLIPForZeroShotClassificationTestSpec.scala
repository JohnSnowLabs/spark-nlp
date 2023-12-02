package com.johnsnowlabs.nlp.annotators.cv

import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotationImage, IAnnotation, ImageAssembler}
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec

class CLIPForZeroShotClassificationTestSpec extends AnyFlatSpec {

  lazy val imageFolder = "src/test/resources/image/"
  lazy val imageDF: DataFrame = ResourceHelper.spark.read
    .format("image")
    .option("dropInvalid", value = true)
    .load(imageFolder)

  lazy val imageAssembler: ImageAssembler = new ImageAssembler()
    .setInputCol("image")
    .setOutputCol("image_assembler")

  lazy val candidateLabels = Array(
    "a photo of a bird",
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a hen",
    "a photo of a hippo",
    "a photo of a room",
    "a photo of a tractor",
    "a photo of an ostrich",
    "a photo of an ox")

  lazy val expected = Map(
    "junco.JPEG" -> "a photo of a bird",
    "egyptian_cat.jpeg" -> "a photo of a cat",
    "palace.JPEG" -> "a photo of a room",
    "hippopotamus.JPEG" -> "a photo of a hippo",
    "hen.JPEG" -> "a photo of a hen",
    "chihuahua.jpg" -> "a photo of a dog",
    "tractor.JPEG" -> "a photo of a tractor",
    "ostrich.JPEG" -> "a photo of an ostrich",
    "ox.JPEG" -> "a photo of an ox",
    "bluetick.jpg" -> "a photo of a dog")

  lazy val model: CLIPForZeroShotClassification = CLIPForZeroShotClassification
    .pretrained()
    .setInputCols("image_assembler")
    .setOutputCol("classification")
    .setCandidateLabels(candidateLabels)
    .setBatchSize(4)

  lazy val pipeline = new Pipeline().setStages(Array(imageAssembler, model))

  behavior of "CLIPForZeroShotClassification"

  private def assertResult(results: DataFrame): Unit = {
    val annotations = Annotation.collect(results, "classification")

    annotations.map(_.head).foreach { case Annotation(_, _, _, result, metadata, _) =>
      val imageFile = metadata("origin").split("/").last
      print(imageFile, result)
      assert(expected(imageFile) == result)
    }
  }

  it should "predict gold standards" taggedAs SlowTest in {
    val results = pipeline.fit(imageDF).transform(imageDF)

    assertResult(results)
  }

  it should "be compatible with LightPipeline" taggedAs SlowTest in {
    val pipelineModel = pipeline.fit(imageDF)
    val lightPipeline = new LightPipeline(pipelineModel)
    val images = expected.keys.map(imageFolder + _).toArray
    val result = lightPipeline.fullAnnotateImage(images)

    result.foreach { row: Map[String, Seq[IAnnotation]] =>
      val imageName =
        row("image_assembler").head.asInstanceOf[AnnotationImage].origin.split("/").last
      val classification = row("classification").head.asInstanceOf[Annotation].result

      assert(classification == expected(imageName))
    }
  }

  it should "be serializable" taggedAs SlowTest in {
    val pipelineModel = pipeline.fit(imageDF)
    pipelineModel.stages.last
      .asInstanceOf[CLIPForZeroShotClassification]
      .write
      .overwrite()
      .save("./tmp_clip_model")

    val loadedModel = CLIPForZeroShotClassification.load("./tmp_clip_model")
    val newPipeline: Pipeline = new Pipeline().setStages(Array(imageAssembler, loadedModel))

    assertResult(newPipeline.fit(imageDF).transform(imageDF))
  }
}
