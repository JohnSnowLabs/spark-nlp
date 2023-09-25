package com.johnsnowlabs.nlp.annotators.cv

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, ImageAssembler}
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Row}
import org.scalatest.flatspec.AnyFlatSpec
class VisionEncoderDecoderForImageCaptioningTestSpec extends AnyFlatSpec {

  lazy val model: VisionEncoderDecoderForImageCaptioning = VisionEncoderDecoderForImageCaptioning
    .pretrained()
    .setInputCols("image_assembler")
    .setOutputCol("caption")
    .setBeamSize(2)
    .setDoSample(false)

  private val imageFolder = "src/test/resources/image/"
  lazy val imageDF: DataFrame = ResourceHelper.spark.read
    .format("image")
    .option("dropInvalid", value = true)
    .load(imageFolder)

  lazy val imageAssembler: ImageAssembler = new ImageAssembler()
    .setInputCol("image")
    .setOutputCol("image_assembler")

  lazy val tokenizer = new Tokenizer()
    .setInputCols("caption")
    .setOutputCol("token")

  behavior of "VisionEncoderDecoderModel"

  correctTranscriber(model, "tf")
  compatibleWithLightPipeline(model, "tf")
  serializableModel(model, "tf")

  def correctTranscriber(
      model: => VisionEncoderDecoderForImageCaptioning,
      engine: => String): Unit = {
    it should "correctly caption" taggedAs SlowTest in {
      val pipeline = new Pipeline().setStages(Array(imageAssembler, model))

      val result = pipeline.fit(imageDF).transform(imageDF)

      result.selectExpr("image.origin", "caption.result[0]").collect().map {
        case Row(_: String, caption: String) => assert(caption.nonEmpty)
      }
    }

    it should s"correctly work with Tokenizer ($engine)" taggedAs SlowTest in {
      val pipeline: Pipeline =
        new Pipeline().setStages(Array(imageAssembler, model, tokenizer))

      val image = imageDF.limit(1)
      val pipelineDF = pipeline.fit(image).transform(image)

      val tokens = Annotation.collect(pipelineDF, "token").head.map(_.getResult)

      assert(tokens.nonEmpty)
    }
  }

  def compatibleWithLightPipeline(
      model: => VisionEncoderDecoderForImageCaptioning,
      engine: => String): Unit = {

    it should s"transform speech to text with LightPipeline ($engine)" taggedAs SlowTest in {
      val pipeline: Pipeline =
        new Pipeline().setStages(Array(imageAssembler, model, tokenizer))

      val pipelineModel = pipeline.fit(imageDF)
      val lightPipeline = new LightPipeline(pipelineModel)
      val result = lightPipeline.fullAnnotateImage(imageFolder + "egyptian_cat.jpeg")

      println(result("token"))
      assert(result("image_assembler").nonEmpty)
      assert(result("caption").nonEmpty)
      assert(result("token").nonEmpty)
    }

    it should s"transform several speeches to text with LightPipeline ($engine)" taggedAs SlowTest in {
      val pipeline: Pipeline =
        new Pipeline().setStages(Array(imageAssembler, model, tokenizer))

      val pipelineModel = pipeline.fit(imageDF)
      val lightPipeline = new LightPipeline(pipelineModel)
      val image = imageFolder + "egyptian_cat.jpeg"
      val results = lightPipeline.fullAnnotateImage(Array(image, image))

      results.foreach { result =>
        assert(result("image_assembler").nonEmpty)
        assert(result("caption").nonEmpty)
        assert(result("token").nonEmpty)
      }
    }
  }

  def serializableModel(
      model: => VisionEncoderDecoderForImageCaptioning,
      engine: => String): Unit = {
    it should s"be serializable ($engine)" taggedAs SlowTest in {

      val pipeline: Pipeline = new Pipeline().setStages(Array(imageAssembler, model))

      val pipelineModel = pipeline.fit(imageDF)
      pipelineModel.stages.last
        .asInstanceOf[VisionEncoderDecoderForImageCaptioning]
        .write
        .overwrite()
        .save("./tmp_visionEncoderDecoder_model")

      val loadedModel =
        VisionEncoderDecoderForImageCaptioning.load("./tmp_visionEncoderDecoder_model")
      val newPipeline: Pipeline = new Pipeline().setStages(Array(imageAssembler, loadedModel))

      newPipeline
        .fit(imageDF)
        .transform(imageDF.limit(1))
        .select("caption")
        .show(truncate = false)
    }
  }

}
