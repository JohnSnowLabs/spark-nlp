package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotationImage, ImageAssembler}
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.TestUtils.measureRAMChange
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, Row}
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable

class AutoGGUFVisionModelTestSpec extends AnyFlatSpec {

  behavior of "AutoGGUFVisionModel"

  lazy val documentAssembler: DocumentAssembler = new DocumentAssembler()
    .setInputCol("caption")
    .setOutputCol("caption_document")

  lazy val imageAssembler: ImageAssembler = new ImageAssembler()
    .setInputCol("image")
    .setOutputCol("image_assembler")

  lazy val imagesPath = "src/test/resources/image/"
  lazy val data: DataFrame = ImageAssembler
    .loadImagesAsBytes(ResourceHelper.spark, imagesPath)
    .withColumn(
      "caption",
      lit("Describe in a short and easy to understand sentence what you see in the image.")
    ) // Add a caption to each image.

  lazy val expectedWords: Map[String, String] = Map(
    "bluetick.jpg" -> "dog",
    "chihuahua.jpg" -> "dog",
    "egyptian_cat.jpeg" -> "cat",
    "hen.JPEG" -> "chick",
    "hippopotamus.JPEG" -> "hippo",
    "junco.JPEG" -> "bird",
    "ostrich.JPEG" -> "ostrich",
    "ox.JPEG" -> "horn",
    "palace.JPEG" -> "room",
    "tractor.JPEG" -> "tractor")

  lazy val nPredict = 40
  lazy val model: AutoGGUFVisionModel = AutoGGUFVisionModel
//    .loadSavedModel(
//      "models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
//      "models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
//      ResourceHelper.spark)
    .pretrained()
    .setInputCols("caption_document", "image_assembler")
    .setOutputCol("completions")
    .setBatchSize(2)
    .setNGpuLayers(99)
    .setNCtx(4096)
    .setMinKeep(0)
    .setMinP(0.05f)
    .setNPredict(nPredict)
    .setPenalizeNl(true)
    .setRepeatPenalty(1.18f)
    .setTemperature(0.05f)
    .setTopK(40)
    .setTopP(0.95f)

  lazy val pipeline: Pipeline =
    new Pipeline().setStages(Array(documentAssembler, imageAssembler, model))

  def checkBinaryContents(): Unit = {
    val imageData = data.select("image.data").limit(1).collect()(0).getAs[Array[Byte]](0)
    val byteContent = data.select("content").limit(1).collect()(0).getAs[Array[Byte]](0)

    assert(imageData.length == byteContent.length)
    assert(imageData sameElements byteContent)
  }

  it should "replace image data with bytes" taggedAs SlowTest in {
    checkBinaryContents()
  }

  it should "caption the images correctly" taggedAs SlowTest in {
    import java.lang.management.ManagementFactory
    val pid = ManagementFactory.getRuntimeMXBean.getName.split("@")(0)
    println(s"Current PID: $pid")

    val result = pipeline.fit(data).transform(data.repartition(1))

    val imageWithCompletions: Array[(AnnotationImage, Annotation)] =
      result.select("image_assembler", "completions").collect().map { row =>
        val image = AnnotationImage(row.getAs[mutable.WrappedArray[Row]](0).head)
        val annotation = Annotation(row.getAs[mutable.WrappedArray[Row]](1).head)
        (image, annotation)
      }

    imageWithCompletions.foreach { case (image, completion) =>
      val fileName = image.origin.split("/").last
      val expectedWord = expectedWords(fileName)
      val wordFound = completion.result.toLowerCase().contains(expectedWord.toLowerCase())
      assert(
        wordFound,
        s"Expected word $expectedWord not found in $completion.result for image $fileName")
    }
  }

  it should "be serializable" taggedAs SlowTest in {
    val pipelineModel = pipeline.fit(data)
    val savePath = "./tmp_autogguf_vision_model"
    pipelineModel.stages.last
      .asInstanceOf[AutoGGUFVisionModel]
      .write
      .overwrite()
      .save(savePath)

    val loadedModel = AutoGGUFVisionModel.load(savePath)
    val newPipeline: Pipeline =
      new Pipeline().setStages(Array(documentAssembler, imageAssembler, loadedModel))

    newPipeline
      .fit(data)
      .transform(data.limit(1))
      .select("completions")
      .show(truncate = false)
  }

  it should "accept protocol prepended paths" taggedAs SlowTest in {
    lazy val pipeline = new Pipeline().setStages(Array(documentAssembler, imageAssembler, model))
    val pipelineModel = pipeline.fit(data)

    val savePath = "file:///tmp/tmp_autoggufvision_model"
    pipelineModel.stages.last
      .asInstanceOf[AutoGGUFVisionModel]
      .write
      .overwrite()
      .save(savePath)

    AutoGGUFVisionModel.load(savePath)
  }

  it should "load models with deprecated parameters" taggedAs SlowTest in {
    AutoGGUFVisionModel.pretrained("llava_v1.5_7b_Q4_0_gguf")
  }

  // This test requires cpu
  it should "be closeable" taggedAs SlowTest in {
    lazy val model: AutoGGUFVisionModel = AutoGGUFVisionModel
      .pretrained()
      .setInputCols("caption_document", "image_assembler")
      .setOutputCol("completions")
      .setNPredict(5)

    pipeline.fit(data).transform(data.limit(1)).show()

    val ramChange = measureRAMChange { model.close() }
    println("Freed RAM after closing the model: " + ramChange + " MB")
    assert(ramChange < -100, "Freed RAM should be greater than 100 MB")
  }

  it should "be able to remove thinking tags" taggedAs SlowTest in {
    val thinkTag = "think"
    val model = AutoGGUFVisionModel
      .loadSavedModel(
        "models/SmolVLM-256M-Instruct-Q8_0.gguf",
        "models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
        ResourceHelper.spark)
      .setInputCols("caption_document", "image_assembler")
      .setOutputCol("completions")
      .setRemoveThinkingTag(thinkTag)
      .setNPredict(500)
      .setTemperature(0.1f)

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, imageAssembler, model))
    val dataThinking = data
      .limit(1)
      .withColumn(
        "caption",
        lit("What is the meaning of life? Think real hard and relate it to the image."))

    dataThinking.select("caption").show(false)

    val result = pipeline.fit(dataThinking).transform(dataThinking)

    val completion = Annotation.collect(result, "completions").flatten.head.result
    println(completion)
    assert(!completion.contains(s"<$thinkTag>") && !completion.contains(s"</$thinkTag>"))
  }
}
