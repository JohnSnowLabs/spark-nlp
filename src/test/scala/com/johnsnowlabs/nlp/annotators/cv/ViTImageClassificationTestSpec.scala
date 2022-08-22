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
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable

class ViTImageClassificationTestSpec extends AnyFlatSpec {

  val imageDF: DataFrame = ResourceHelper.spark.read
    .format("image")
    .option("dropInvalid", value = true)
    .load("src/test/resources/image/")

  val imageAssembler: ImageAssembler = new ImageAssembler()
    .setInputCol("image")
    .setOutputCol("image_assembler")

  val imageClassifier: ViTForImageClassification = ViTForImageClassification
    .pretrained()
    .setInputCols("image_assembler")
    .setOutputCol("class")

  val pipeline: Pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))

  "ViTForImageClassification" should "predict correct ImageNet classes" taggedAs SlowTest in {

    val pipelineDF = pipeline.fit(imageDF).transform(imageDF)

    val goldStandards =
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

    val predictedResults = pipelineDF
      .select("class.result", "image.origin")
      .rdd
      .flatMap(row =>
        Map(
          row.getAs[mutable.WrappedArray[String]](0)(0) ->
            row.getString(1).split("/").last))
      .collect()

    predictedResults.foreach { x =>
      val goldValue = goldStandards(x._2)
      val predictValue = x._1
      assert(goldValue === predictValue)
    }

  }

  "ViTForImageClassification" should "be serializable" taggedAs SlowTest in {

    val pipelineModel = pipeline.fit(imageDF)
    val pipelineDF = pipelineModel.transform(imageDF)
    pipelineDF.take(1)

    pipelineModel.stages.last
      .asInstanceOf[ViTForImageClassification]
      .write
      .overwrite()
      .save("./tmp_ViTModel")

    // load the saved ViTForImageClassification model
    val loadedViTModel = ViTForImageClassification.load("./tmp_ViTModel")

    val loadedPipeline = new Pipeline().setStages(Array(imageAssembler, loadedViTModel))
    val loadedPipelineModel = loadedPipeline.fit(imageDF)
    val loadedPipelineModelDF = loadedPipelineModel.transform(imageDF)

    loadedPipelineModelDF.select("class.result", "image_assembler.origin").show(3, truncate = 120)

    // save the whole pipeline
    loadedPipelineModelDF.write.mode("overwrite").parquet("./tmp_vit_pipeline")

    // load the whole pipeline
    val loadedProcessedPipelineDF = ResourceHelper.spark.read.parquet("./tmp_vit_pipeline")
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

  "ViTForImageClassification" should "benchmark" taggedAs SlowTest in {

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

  "ViTForImageClassification" should "work with LightPipeline" taggedAs FastTest in {
    val pipelineModel = pipeline.fit(imageDF)
    val lightPipeline = new LightPipeline(pipelineModel)

    val prediction = lightPipeline.fullAnnotateImage("src/test/resources/image/junco.JPEG")

    assert(prediction("image_assembler").nonEmpty)
    assert(prediction("class").nonEmpty)
  }

}
