/*
 * Copyright 2017-2024 John Snow Labs
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

import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations, ImageAssembler}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.lit
import org.scalatest.flatspec.AnyFlatSpec

class Gemma3ForMultiModalTestSpec extends AnyFlatSpec {

  lazy val model = getGemma3ForMultiModalPipelineModel

  "Gemma3ForMultiModal" should "answer a question for a given image" taggedAs SlowTest in {

    val testDF = getTestDF
    val result = model.transform(testDF)

    val answerAnnotation = AssertAnnotations.getActualResult(result, "answer")

    answerAnnotation.foreach { annotation =>
      annotation.foreach(a => assert(a.result.nonEmpty))
    }

    answerAnnotation.foreach { annotation =>
      annotation.foreach(a => println(a.result))
    }

  }

  it should "work with light pipeline annotate" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/image/egyptian_cat.jpeg"
    val resultAnnotate =
      lightPipeline.annotate(
        imagePath,
        "<bos><start_of_turn>user\nYou are a helpful assistant.\n\n<start_of_image>Describe this image in detail.<end_of_turn>\n<start_of_turn>model\n")

    assert(
      resultAnnotate("answer").head.contains("cat") ||
        resultAnnotate("answer").head.contains("kitten"))
  }

  it should "work with light pipeline full annotate" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/image/bluetick.jpg"
    val resultFullAnnotate =
      lightPipeline.fullAnnotateImage(
        imagePath,
        "<bos><start_of_turn>user\nYou are a helpful assistant.\n\n<start_of_image>Describe this image in detail.<end_of_turn>\n<start_of_turn>model\n")

    val answerAnnotation = resultFullAnnotate("answer").head.asInstanceOf[Annotation]

    println(s"imageName.result: ${answerAnnotation.result}")
    assert(answerAnnotation.result.nonEmpty)
  }

  it should "fullAnnotate with empty Map when a text is empty" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagesPath = Array(
      "src/test/resources/image/bluetick.jpg",
      "src/test/resources/image/chihuahua.jpg",
      "src/test/resources/image/egyptian_cat.jpeg")
    val question =
      "<bos><start_of_turn>user\nYou are a helpful assistant.\n\n<start_of_image>Describe this image in detail.<end_of_turn>\n<start_of_turn>model\n"
    val questions = Array(question, "", question)

    val resultFullAnnotate = lightPipeline.fullAnnotateImages(imagesPath, questions)

    resultFullAnnotate.zip(imagesPath).foreach { case (annotateMap, imagePath) =>
      imagePath match {
        case "src/test/resources/image/chihuahua.jpg" =>
          // For the chihuahua image, the annotateMap should be empty because the question is empty
          assert(
            annotateMap.nonEmpty,
            s"Expected empty map for image: $imagePath, but got: $annotateMap")

        case _ =>
          assert(annotateMap.nonEmpty, s"Expected non-empty map for image: $imagePath")

          annotateMap.get("answer") match {
            case Some(annotations) =>
              annotations.foreach { iAnnotation =>
                val annotation = iAnnotation.asInstanceOf[Annotation]
                assert(
                  annotation.result.nonEmpty,
                  s"Expected non-empty result for image: $imagePath, but got empty result")
              }
            case None =>
              fail(s"'answer' key not found in annotateMap for image: $imagePath")
          }
      }
    }
  }

  it should "annotate with empty Map when a text is empty" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagesPath = Array(
      "src/test/resources/image/bluetick.jpg",
      "src/test/resources/image/chihuahua.jpg",
      "src/test/resources/image/egyptian_cat.jpeg")
    val question =
      "<bos><start_of_turn>user\nYou are a helpful assistant.\n\n<start_of_image>Describe this image in detail.<end_of_turn>\n<start_of_turn>model\n"
    val questions = Array(question, "", question)

    val resultAnnotate = lightPipeline.annotate(imagesPath, questions)

    resultAnnotate.foreach { annotate =>
      println(s"annotate: $annotate")
    }

    resultAnnotate.zip(imagesPath).foreach { case (annotateMap, imagePath) =>
      imagePath match {
        case "src/test/resources/image/chihuahua.jpg" =>
          // For the chihuahua image, the annotateMap should be empty because the question is empty
          assert(
            annotateMap.nonEmpty,
            s"Expected empty map for image: $imagePath, but got: $annotateMap")

        case _ =>
          assert(annotateMap.nonEmpty, s"Expected non-empty map for image: $imagePath")

          annotateMap.get("answer") match {
            case Some(annotations) =>
              annotations.foreach { annotation =>
                assert(
                  annotation.nonEmpty,
                  s"Expected non-empty result for image: $imagePath, but got empty result")
              }
            case None =>
              fail(s"'answer' key not found in annotateMap for image: $imagePath")
          }
      }
    }

  }

  private def getGemma3ForMultiModalPipelineModel = {
    val testDF = getTestDF

    val imageAssembler: ImageAssembler = new ImageAssembler()
      .setInputCol("image")
      .setOutputCol("image_assembler")

    val loadModel = Gemma3ForMultiModal
      .pretrained()
      .setInputCols("image_assembler")
      .setOutputCol("answer")
      .setMaxOutputLength(50)

    val newPipeline: Pipeline =
      new Pipeline().setStages(Array(imageAssembler, loadModel))

    val pipelineModel = newPipeline.fit(testDF)

    pipelineModel
      .transform(testDF)
      .show(truncate = false)

    pipelineModel
      .transform(testDF)
      .show(truncate = false)

    pipelineModel.stages.last
      .asInstanceOf[Gemma3ForMultiModal]
      .write
      .overwrite()
      .save("/tmp/Gemma3-4b-4bit-model")

    val loadedLLAMA3 = Gemma3ForMultiModal.load("/tmp/Gemma3-4b-4bit-model")

    val loadedPipeline = new Pipeline().setStages(Array(imageAssembler, loadedLLAMA3))

    loadedPipeline
      .fit(testDF)
      .transform(testDF)
      .show(truncate = false)

    pipelineModel
  }

  private def getTestDF: DataFrame = {
    val imageFolder = "src/test/resources/image/"
    val imageDF: DataFrame = ResourceHelper.spark.read
      .format("image")
      .option("dropInvalid", value = true)
      .load(imageFolder)

    val testDF: DataFrame = imageDF.withColumn(
      "text",
      lit(
        "<bos><start_of_turn>user\nYou are a helpful assistant.\n\n<start_of_image>Describe this image in detail.<end_of_turn>\n<start_of_turn>model\n"))

    testDF
  }

}
