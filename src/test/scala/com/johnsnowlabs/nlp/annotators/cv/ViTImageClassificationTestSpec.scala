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

package com.johnsnowlabs.nlp.annotators.cv

import com.johnsnowlabs.nlp.ImageAssembler
import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable

trait ViTForImageClassificationBehaviors { this: AnyFlatSpec =>
  val imageDF: DataFrame = ResourceHelper.spark.read
    .format("image")
    .option("dropInvalid", value = true)
    .load("src/test/resources/image/")

  val imageAssembler: ImageAssembler = new ImageAssembler()
    .setInputCol("image")
    .setOutputCol("image_assembler")

  def behaviorsViTForImageClassification[M <: ViTForImageClassification](
      loadModelFunction: => String => M,
      vitClassifier: => ViTForImageClassification,
      expectedPredictions: => Map[String, String]): Unit = {

    def setUpImageClassifierPipeline(): Pipeline = {
      val imageClassifier: ViTForImageClassification = vitClassifier
        .setInputCols("image_assembler")
        .setOutputCol("class")

      val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))
      pipeline
    }

    it should "predict correct ImageNet classes" taggedAs SlowTest in {

      val pipeline = setUpImageClassifierPipeline()
      val pipelineDF = pipeline.fit(imageDF).transform(imageDF)

      assertPredictions(pipelineDF, expectedPredictions)

    }

    it should "be serializable" taggedAs SlowTest in {

      val pipeline = setUpImageClassifierPipeline()
      val pipelineModel = pipeline.fit(imageDF)
      val pipelineDF = pipelineModel.transform(imageDF)
      pipelineDF.take(1)

      val classifierClass = vitClassifier.getClass.toString.split("\\.").last
      val tmpSavedFolder = s"./tmp_$classifierClass"
      pipelineModel.stages.last
        .asInstanceOf[M]
        .write
        .overwrite()
        .save(tmpSavedFolder)

      // load the saved ViTForImageClassification model
      val loadedViTModel = loadModelFunction(tmpSavedFolder)

      val loadedPipeline = new Pipeline().setStages(Array(imageAssembler, loadedViTModel))
      val loadedPipelineModel = loadedPipeline.fit(imageDF)
      val loadedPipelineModelDF = loadedPipelineModel.transform(imageDF)

      assertPredictions(loadedPipelineModelDF, expectedPredictions)

      loadedPipelineModelDF
        .select("class.result", "image_assembler.origin")
        .show(3, truncate = 120)

      // save the whole pipeline
      val tmpPipelinePath = tmpSavedFolder + "_pipeline"
      loadedPipelineModelDF.write.mode("overwrite").parquet(tmpPipelinePath)

      // load the whole pipeline
      val loadedProcessedPipelineDF = ResourceHelper.spark.read.parquet(tmpPipelinePath)
      loadedProcessedPipelineDF
        .select("class.result", "image_assembler.origin")
        .show(3, truncate = 120)

      loadedProcessedPipelineDF
        .select("image_assembler")
        .show(1, truncate = 120)

      loadedProcessedPipelineDF
        .select("image_assembler")
        .printSchema()

    }

    it should "benchmark" taggedAs SlowTest in {

      val imageClassifier: ViTForImageClassification = vitClassifier
        .setInputCols("image_assembler")
        .setOutputCol("class")

      val pipeline: Pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))

      Array(2, 4, 8, 16, 32, 128).foreach(b => {
        imageClassifier.setBatchSize(b)

        val pipelineModel = pipeline.fit(imageDF)
        val pipelineDF = pipelineModel.transform(imageDF)

        println(
          s"batch size: ${pipelineModel.stages.last.asInstanceOf[ViTForImageClassification].getBatchSize}")

        Benchmark.measure(
          iterations = 5,
          forcePrint = true,
          description = "Time to save pipeline") {
          pipelineDF.write.mode("overwrite").parquet("./tmp_vit_image_classifier")
        }
      })
    }

    it should "work with LightPipeline" taggedAs SlowTest in {

      val pipeline = setUpImageClassifierPipeline()
      val pipelineModel = pipeline.fit(imageDF)
      val lightPipeline = new LightPipeline(pipelineModel)

      val prediction = lightPipeline.fullAnnotateImage("src/test/resources/image/junco.JPEG")

      assert(prediction("image_assembler").nonEmpty)
      assert(prediction("class").nonEmpty)
    }

    it should "return empty result when image path is wrong with LightPipeline" taggedAs SlowTest in {

      val pipeline = setUpImageClassifierPipeline()
      val pipelineModel = pipeline.fit(imageDF)
      val lightPipeline = new LightPipeline(pipelineModel)

      val prediction = lightPipeline.fullAnnotateImage("./image")

      assert(prediction("image_assembler").isEmpty)
      assert(prediction("class").isEmpty)

      val images =
        Array("src/test/resources/image/hen.JPEG", "src/test/resources/image/missing_file.mf")
      val predictions = lightPipeline.fullAnnotateImages(images)

      assert(predictions(0)("image_assembler").nonEmpty)
      assert(predictions(0)("class").nonEmpty)
      assert(predictions(1)("image_assembler").isEmpty)
      assert(predictions(1)("class").isEmpty)

      val predictionsFullAnnotate = lightPipeline.fullAnnotate(images)
      assert(predictionsFullAnnotate(0)("image_assembler").nonEmpty)
      assert(predictionsFullAnnotate(0)("class").nonEmpty)
      assert(predictionsFullAnnotate(1)("image_assembler").isEmpty)
      assert(predictionsFullAnnotate(1)("class").isEmpty)
    }

    it should "work for mix inputs" taggedAs SlowTest in {
      val pipeline = setUpImageClassifierPipeline()
      val pipelineModel = pipeline.fit(imageDF)
      val lightPipeline = new LightPipeline(pipelineModel)

      val prediction = lightPipeline.fullAnnotateImage("./image")

      assert(prediction("image_assembler").isEmpty)
      assert(prediction("class").isEmpty)

      val images =
        Array("src/test/resources/image/hen.JPEG", "this is a text")
      val predictions = lightPipeline.fullAnnotateImages(images)

      assert(predictions(0)("image_assembler").nonEmpty)
      assert(predictions(0)("class").nonEmpty)
      assert(predictions(1)("image_assembler").isEmpty)
      assert(predictions(1)("class").isEmpty)
    }

  }

  private def assertPredictions[M <: ViTForImageClassification](
      pipelineDF: DataFrame,
      expectedPredictions: Map[String, String]): Unit = {
    val predictedResults = pipelineDF
      .select("class.result", "image.origin")
      .rdd
      .flatMap(row =>
        Map(
          row.getAs[mutable.WrappedArray[String]](0)(0) ->
            row.getString(1).split("/").last))
      .collect()

    predictedResults.foreach { x =>
      val goldValue = expectedPredictions(x._2)
      val predictValue = x._1
      assert(goldValue === predictValue)
    }
  }
}

class ViTImageClassificationTestSpec extends AnyFlatSpec with ViTForImageClassificationBehaviors {

  behavior of "ViTForImageClassification"

  lazy val goldStandards: Map[String, String] =
    Map(
      "palace.JPEG" -> "palace",
      "egyptian_cat.jpeg" -> "Egyptian cat",
      "hippopotamus.JPEG" -> "hippopotamus, hippo, river horse, Hippopotamus amphibius",
      "hen.JPEG" -> "hen",
      "ostrich.JPEG" -> "ostrich, Struthio camelus",
      "junco.JPEG" -> "junco, snowbird",
      "bluetick.jpg" -> "bluetick",
      "chihuahua.jpg" -> "Chihuahua",
      "tractor.JPEG" -> "tractor",
      "ox.JPEG" -> "ox")

  private lazy val model: ViTForImageClassification = ViTForImageClassification.pretrained()

  it should behave like
    behaviorsViTForImageClassification[ViTForImageClassification](
      ViTForImageClassification.load,
      model,
      goldStandards)
}
