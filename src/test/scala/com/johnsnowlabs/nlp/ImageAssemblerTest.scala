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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.IMAGE
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File

class ImageAssemblerTest extends AnyFlatSpec {

  val imagesPath = "src/test/resources/image/"
  val dataFrame: DataFrame = ResourceHelper.spark.read
    .format("image")
    .load(imagesPath)

  val emptyDF: DataFrame = ResourceHelper.spark.emptyDataFrame

  "ImageAssembler" should "assemble an image input" taggedAs FastTest in {

    val imageAssembler = new ImageAssembler()
      .setInputCol("image")
      .setOutputCol("image_assembler")

    val assembled = imageAssembler.transform(dataFrame)

    val result = AssertAnnotations.getActualImageResult(assembled, "image_assembler")

    assert(result.nonEmpty)

    result.foreach(annotationImages =>
      annotationImages.foreach { annotationImage =>
        assert(annotationImage.annotatorType == IMAGE)
        assert(annotationImage.origin.contains(imagesPath))
        assert(annotationImage.height >= 0)
        assert(annotationImage.width >= 0)
        assert(annotationImage.nChannels >= 0)
        assert(annotationImage.mode >= 0)
        assert(annotationImage.result.nonEmpty)
        assert(annotationImage.metadata.nonEmpty)
      })
  }

  it should "work with LightPipeline with one image file" taggedAs FastTest in {
    val imageAssembler = new ImageAssembler()
      .setInputCol("image")
      .setOutputCol("image_assembler")

    val pipeline: Pipeline = new Pipeline().setStages(Array(imageAssembler))
    val pipelineModel = pipeline.fit(emptyDF)
    val lightPipeline = new LightPipeline(pipelineModel)
    val result = lightPipeline.fullAnnotateImage("src/test/resources/image/hippopotamus.JPEG")

    assert(result("image_assembler").nonEmpty)

  }

  it should "work with LightPipeline with many image files" taggedAs FastTest in {
    val imageAssembler = new ImageAssembler()
      .setInputCol("image")
      .setOutputCol("image_assembler")
    val images =
      Array("src/test/resources/image/bluetick.jpg", "src/test/resources/image/hen.JPEG")

    val pipeline: Pipeline = new Pipeline().setStages(Array(imageAssembler))
    val pipelineModel = pipeline.fit(emptyDF)
    val lightPipeline = new LightPipeline(pipelineModel)
    val result = lightPipeline.fullAnnotateImage(images)

    assert(result.length == images.length)
    result.foreach(annotation => assert(annotation("image_assembler").nonEmpty))

  }

}
