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
import com.johnsnowlabs.nlp.{Annotation, AnnotationImage, AssertAnnotations, ImageAssembler}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.lit
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers._
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
import java.io.{File, FileOutputStream}

class JanusForMultiModalTestSpec extends AnyFlatSpec {

  def reshape2D(data: Array[Float], rows: Int, cols: Int): Array[Array[Float]] = {
    data.grouped(cols).toArray.map(_.toArray)
  }

  def reshape3D(
      data: Array[Float],
      depth: Int,
      rows: Int,
      cols: Int): Array[Array[Array[Float]]] = {
    data.grouped(rows * cols).toArray.map { slice =>
      reshape2D(slice, rows, cols)
    }
  }

  def reshape4D(
      data: Array[Float],
      batch: Int,
      depth: Int,
      rows: Int,
      cols: Int): Array[Array[Array[Array[Float]]]] = {
    data.grouped(depth * rows * cols).toArray.map { slice =>
      reshape3D(slice, depth, rows, cols)
    }
  }
  lazy val model = getJanusForMultiModalPipelineModel

  "JanusForMultiModal" should "answer a question for a given image" taggedAs SlowTest in {

    val testDF = getTestDF
    val result = model.transform(testDF)

    result.printSchema()
    val answerAnnotation = AssertAnnotations.getActualResult(result, "answer")

    answerAnnotation.foreach { annotation =>
      annotation.foreach(a => assert(a.result.nonEmpty))
    }

    answerAnnotation.foreach { annotation =>
      annotation.foreach(a => println(a.result))
    }

  }
  "reshape2D" should "reshape a 1D array into a 2D array" taggedAs SlowTest in {
    val data = Array(1f, 2f, 3f, 4f, 5f, 6f)
    val rows = 2
    val cols = 3
    val expected = Array(Array(1f, 2f, 3f), Array(4f, 5f, 6f))
    reshape2D(data, rows, cols) shouldEqual expected
  }

  "reshape3D" should "reshape a 1D array into a 3D array" taggedAs SlowTest in {
    val data = Array(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)
    val depth = 2
    val rows = 2
    val cols = 3
    val expected = Array(
      Array(Array(1f, 2f, 3f), Array(4f, 5f, 6f)),
      Array(Array(7f, 8f, 9f), Array(10f, 11f, 12f)))
    reshape3D(data, depth, rows, cols) shouldBe expected
  }

  it should "generate images when generate image mode is set to true" taggedAs SlowTest in {
    model.stages.last.asInstanceOf[JanusForMultiModal].setImageGenerateMode(true)
    model.stages.last.asInstanceOf[JanusForMultiModal].setRandomSeed(123467L)
    model.stages.last.asInstanceOf[JanusForMultiModal].setNumOfParallelImages(1)
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/image1.jpg"
    val resultAnnotate =
      lightPipeline.fullAnnotateImage(
        imagePath,
        "User: A close-up professional photo of Yorkshire Terrier on beach, extremely detailed, hyper realistic, full hd resolution, with a blurred background. The dog is looking at the camera, with a curious expression, and its fur is shiny and well-groomed. The beach is sandy, with gentle waves lapping at the shore, and a clear blue sky overhead. The lighting is soft and natural, casting a warm glow over the scene. The overall mood is peaceful and serene, capturing a moment of quiet contemplation and connection with nature.\n\nAssistant:")
//        "User: Create a detailed image of a whimsical forest filled with vibrant, oversized mushrooms, glowing flowers, and towering, twisted trees with bioluminescent vines. The atmosphere is magical, with soft, ethereal light filtering through a misty canopy. Small floating orbs of light hover among the branches, and tiny fairy-like creatures flit through the air. A winding, moss-covered path leads to a mysterious glowing portal hidden within the trees. The scene should feel enchanting, otherworldly, and full of wonder, like a dreamlike fantasy realm.\n\nAssistant:")

    val answerAnnotation = resultAnnotate("answer").head.asInstanceOf[Annotation]
    println(s"imageName.result: ${answerAnnotation.result}")

    // generated image should be in the metadata as a base64 string with the keys "generated_image_0", "generated_image_1", etc.
    // find the keys that contain the generated images
    val generatedImageKeys = answerAnnotation.metadata.keys.filter(_.contains("generated_image"))

    assert(generatedImageKeys.nonEmpty)

    for (key <- generatedImageKeys) {
      val generatedImage = answerAnnotation.metadata(key).asInstanceOf[String]
      val decodedImage =
        java.util.Base64.getDecoder.decode(generatedImage)
      // save the image to the disk
      val fos =
        new FileOutputStream(new File(s"src/test/resources/images/generated_image_$key.jpg"))
      fos.write(decodedImage)
      fos.close()
    }
  }

  it should "work with light pipeline annotate" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/image1.jpg"
    val resultAnnotate =
      lightPipeline.annotate(
        imagePath,
        "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: <image_placeholder>Describe image in details\n\nAssistant:")
    println(s"resultAnnotate: $resultAnnotate")

    assert(resultAnnotate("answer").head.contains("cat"))
  }

  it should "work with light pipeline full annotate" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/bluetick.jpg"
    val resultFullAnnotate =
      lightPipeline.fullAnnotateImage(
        imagePath,
        "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: <image_placeholder>Describe image in details\n\nAssistant:")

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
      "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: <image_placeholder>Describe image in details\n\nAssistant:"
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
      "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: <image_placeholder>Describe image in details\n\nAssistant:"
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

  private def getJanusForMultiModalPipelineModel = {
    val testDF = getTestDF

    val imageAssembler: ImageAssembler = new ImageAssembler()
      .setInputCol("image")
      .setOutputCol("image_assembler")

    val loadModel = JanusForMultiModal
      .pretrained()
      .setInputCols("image_assembler")
      .setOutputCol("answer")
      .setMaxOutputLength(50)

    val newPipeline: Pipeline =
      new Pipeline().setStages(Array(imageAssembler, loadModel))

    newPipeline.fit(testDF)
  }

  private def getTestDF: DataFrame = {
    val imageFolder = "src/test/resources/images/"
    val imageDF: DataFrame = ResourceHelper.spark.read
      .format("image")
      .option("dropInvalid", value = true)
      .load(imageFolder)

    val testDF: DataFrame = imageDF.withColumn(
      "text",
      lit(
        "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: <image_placeholder>Describe image in details\n\nAssistant:"))

    testDF
  }

}
