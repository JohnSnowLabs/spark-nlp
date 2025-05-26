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

import java.io.{File, FileOutputStream}

class Florence2TransformerTestSpec extends AnyFlatSpec {

  lazy val model = getFlorence2TransformerPipelineModel

  "Florence2Transformer" should "answer a question for a given image" taggedAs SlowTest in {

    val testDF = getTestDF
    val result = model.transform(testDF)

    var answerAnnotation = AssertAnnotations.getActualResult(result, "answer")

    answerAnnotation.foreach { annotation =>
      annotation.foreach(a => assert(a.result.nonEmpty))
    }

    answerAnnotation.foreach { annotation =>
      annotation.foreach(a => println(a.result))
    }

    answerAnnotation.foreach { annotations =>
      for (annotation <- annotations) {
        if (annotation.metadata.contains("florence2_image")) {
          val florence2ImageBase64 = annotation.metadata("florence2_image")
          val decodedFlorence2Image = java.util.Base64.getDecoder.decode(florence2ImageBase64)
          val fos = new FileOutputStream(
            new File(
              s"src/test/resources/images/florence2_image_${System.currentTimeMillis()}.png"))
          fos.write(decodedFlorence2Image)
          fos.close()
        }

        if (annotation.metadata.contains("florence2_postprocessed_raw")) {
          val florence2PostprocessedRaw = annotation.metadata("florence2_postprocessed_raw")
          println(s"florence2PostprocessedRaw: $florence2PostprocessedRaw")
        }
      }
    }
  }

  it should "work with light pipeline annotate" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/image1.jpg"
    val resultAnnotate =
      lightPipeline.annotate(
        imagePath,
        "<s>Locate the objects with category name in the image.</s>")
    println(s"resultAnnotate: $resultAnnotate")

    assert(resultAnnotate("answer").head.contains("box"))
  }

  it should "work with light pipeline full annotate" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/bluetick.jpg"
    val resultFullAnnotate =
      lightPipeline.fullAnnotateImage(
        imagePath,
        "<s>Locate the objects with category name in the image.</s>")

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
      "<s>Locate the objects with category name in the image.</s>"
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
      "<s>Locate the objects with category name in the image.</s>"
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

  it should "run OCR task (<OCR>)" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/ocr_sample.jpg"
    val result = lightPipeline.fullAnnotateImage(imagePath, "<OCR>")
    val answer = result("answer").head.asInstanceOf[Annotation]
    if (answer.metadata.contains("florence2_image")) {
      val florence2ImageBase64 = answer.metadata("florence2_image")
      val decodedFlorence2Image = java.util.Base64.getDecoder.decode(florence2ImageBase64)
      val fos = new FileOutputStream(
        new File(s"src/test/resources/images/florence2_image_${System.currentTimeMillis()}.png"))
      fos.write(decodedFlorence2Image)
      fos.close()
    }
    if (answer.metadata.contains("florence2_postprocessed_raw")) {
      val florence2PostprocessedRaw = answer.metadata("florence2_postprocessed_raw")
      println(s"florence2PostprocessedRaw: $florence2PostprocessedRaw")
    }
    println(s"<OCR> result: ${answer.result}")
    assert(answer.result.nonEmpty)
  }

  it should "run OCR with region task (<OCR_WITH_REGION>)" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/ocr_sample.jpg"
    val result = lightPipeline.fullAnnotateImage(imagePath, "<OCR_WITH_REGION>")
    val answer = result("answer").head.asInstanceOf[Annotation]
    if (answer.metadata.contains("florence2_image")) {
      val florence2ImageBase64 = answer.metadata("florence2_image")
      val decodedFlorence2Image = java.util.Base64.getDecoder.decode(florence2ImageBase64)
      val fos = new FileOutputStream(
        new File(s"src/test/resources/images/florence2_image_${System.currentTimeMillis()}.png"))
      fos.write(decodedFlorence2Image)
      fos.close()
    }
    if (answer.metadata.contains("florence2_postprocessed_raw")) {
      val florence2PostprocessedRaw = answer.metadata("florence2_postprocessed_raw")
      println(s"florence2PostprocessedRaw: $florence2PostprocessedRaw")
    }
    println(s"<OCR_WITH_REGION> result: ${answer.result}")
    assert(answer.result.nonEmpty)
  }

  it should "run image captioning tasks (<CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>)" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/car.jpg"
    val prompts = Seq("<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>")
    prompts.foreach { prompt =>
      val result = lightPipeline.fullAnnotateImage(imagePath, prompt)
      val answer = result("answer").head.asInstanceOf[Annotation]
      if (answer.metadata.contains("florence2_image")) {
        val florence2ImageBase64 = answer.metadata("florence2_image")
        val decodedFlorence2Image = java.util.Base64.getDecoder.decode(florence2ImageBase64)
        val fos = new FileOutputStream(
          new File(
            s"src/test/resources/images/florence2_image_${System.currentTimeMillis()}.png"))
        fos.write(decodedFlorence2Image)
        fos.close()
      }
      if (answer.metadata.contains("florence2_postprocessed_raw")) {
        val florence2PostprocessedRaw = answer.metadata("florence2_postprocessed_raw")
        println(s"florence2PostprocessedRaw: $florence2PostprocessedRaw")
      }
      println(s"$prompt result: ${answer.result}")
      assert(answer.result.nonEmpty)
    }
  }

  it should "run object detection (<OD>) and dense region caption (<DENSE_REGION_CAPTION>)" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/car.jpg"
    val prompts = Seq("<OD>", "<DENSE_REGION_CAPTION>")
    prompts.foreach { prompt =>
      val result = lightPipeline.fullAnnotateImage(imagePath, prompt)
      val answer = result("answer").head.asInstanceOf[Annotation]
      if (answer.metadata.contains("florence2_image")) {
        val florence2ImageBase64 = answer.metadata("florence2_image")
        val decodedFlorence2Image = java.util.Base64.getDecoder.decode(florence2ImageBase64)
        val fos = new FileOutputStream(
          new File(
            s"src/test/resources/images/florence2_image_${System.currentTimeMillis()}.png"))
        fos.write(decodedFlorence2Image)
        fos.close()
      }
      if (answer.metadata.contains("florence2_postprocessed_raw")) {
        val florence2PostprocessedRaw = answer.metadata("florence2_postprocessed_raw")
        println(s"florence2PostprocessedRaw: $florence2PostprocessedRaw")
      }
      println(s"$prompt result: ${answer.result}")
      assert(answer.result.nonEmpty)
    }
  }

  it should "run region proposal (<REGION_PROPOSAL>)" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/car.jpg"
    val result = lightPipeline.fullAnnotateImage(imagePath, "<REGION_PROPOSAL>")
    val answer = result("answer").head.asInstanceOf[Annotation]
    if (answer.metadata.contains("florence2_image")) {
      val florence2ImageBase64 = answer.metadata("florence2_image")
      val decodedFlorence2Image = java.util.Base64.getDecoder.decode(florence2ImageBase64)
      val fos = new FileOutputStream(
        new File(s"src/test/resources/images/florence2_image_${System.currentTimeMillis()}.png"))
      fos.write(decodedFlorence2Image)
      fos.close()
    }
    if (answer.metadata.contains("florence2_postprocessed_raw")) {
      val florence2PostprocessedRaw = answer.metadata("florence2_postprocessed_raw")
      println(s"florence2PostprocessedRaw: $florence2PostprocessedRaw")
    }
    println(s"<REGION_PROPOSAL> result: ${answer.result}")
    assert(answer.result.nonEmpty)
  }

  it should "run phrase grounding (<CAPTION_TO_PHRASE_GROUNDING> car)" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/car.jpg"
    val result = lightPipeline.fullAnnotateImage(imagePath, "<CAPTION_TO_PHRASE_GROUNDING> car")
    val answer = result("answer").head.asInstanceOf[Annotation]
    if (answer.metadata.contains("florence2_image")) {
      val florence2ImageBase64 = answer.metadata("florence2_image")
      val decodedFlorence2Image = java.util.Base64.getDecoder.decode(florence2ImageBase64)
      val fos = new FileOutputStream(
        new File(s"src/test/resources/images/florence2_image_${System.currentTimeMillis()}.png"))
      fos.write(decodedFlorence2Image)
      fos.close()
    }
    if (answer.metadata.contains("florence2_postprocessed_raw")) {
      val florence2PostprocessedRaw = answer.metadata("florence2_postprocessed_raw")
      println(s"florence2PostprocessedRaw: $florence2PostprocessedRaw")
    }
    println(s"<CAPTION_TO_PHRASE_GROUNDING> result: ${answer.result}")
    assert(answer.result.nonEmpty)
  }

  it should "run referring expression segmentation (<REFERRING_EXPRESSION_SEGMENTATION> car)" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/car.jpg"
    val result =
      lightPipeline.fullAnnotateImage(imagePath, "<REFERRING_EXPRESSION_SEGMENTATION> car")
    val answer = result("answer").head.asInstanceOf[Annotation]
    if (answer.metadata.contains("florence2_image")) {
      val florence2ImageBase64 = answer.metadata("florence2_image")
      val decodedFlorence2Image = java.util.Base64.getDecoder.decode(florence2ImageBase64)
      val fos = new FileOutputStream(
        new File(s"src/test/resources/images/florence2_image_${System.currentTimeMillis()}.png"))
      fos.write(decodedFlorence2Image)
      fos.close()
    }
    if (answer.metadata.contains("florence2_postprocessed_raw")) {
      val florence2PostprocessedRaw = answer.metadata("florence2_postprocessed_raw")
      println(s"florence2PostprocessedRaw: $florence2PostprocessedRaw")
    }
    println(s"<REFERRING_EXPRESSION_SEGMENTATION> result: ${answer.result}")
    assert(answer.result.nonEmpty)
  }

  it should "run region to segmentation (<REGION_TO_SEGMENTATION> region1)" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/car.jpg"
    val result = lightPipeline.fullAnnotateImage(imagePath, "<REGION_TO_SEGMENTATION> region1")
    val answer = result("answer").head.asInstanceOf[Annotation]
    if (answer.metadata.contains("florence2_image")) {
      val florence2ImageBase64 = answer.metadata("florence2_image")
      val decodedFlorence2Image = java.util.Base64.getDecoder.decode(florence2ImageBase64)
      val fos = new FileOutputStream(
        new File(s"src/test/resources/images/florence2_image_${System.currentTimeMillis()}.png"))
      fos.write(decodedFlorence2Image)
      fos.close()
    }
    if (answer.metadata.contains("florence2_postprocessed_raw")) {
      val florence2PostprocessedRaw = answer.metadata("florence2_postprocessed_raw")
      println(s"florence2PostprocessedRaw: $florence2PostprocessedRaw")
    }
    println(s"<REGION_TO_SEGMENTATION> result: ${answer.result}")
    assert(answer.result.nonEmpty)
  }

  it should "run open vocabulary detection (<OPEN_VOCABULARY_DETECTION> car)" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/car.jpg"
    val result = lightPipeline.fullAnnotateImage(imagePath, "<OPEN_VOCABULARY_DETECTION> car")
    val answer = result("answer").head.asInstanceOf[Annotation]
    if (answer.metadata.contains("florence2_image")) {
      val florence2ImageBase64 = answer.metadata("florence2_image")
      val decodedFlorence2Image = java.util.Base64.getDecoder.decode(florence2ImageBase64)
      val fos = new FileOutputStream(
        new File(s"src/test/resources/images/florence2_image_${System.currentTimeMillis()}.png"))
      fos.write(decodedFlorence2Image)
      fos.close()
    }
    if (answer.metadata.contains("florence2_postprocessed_raw")) {
      val florence2PostprocessedRaw = answer.metadata("florence2_postprocessed_raw")
      println(s"florence2PostprocessedRaw: $florence2PostprocessedRaw")
    }
    println(s"<OPEN_VOCABULARY_DETECTION> result: ${answer.result}")
    assert(answer.result.nonEmpty)
  }

  it should "run region to category (<REGION_TO_CATEGORY> region1)" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/car.jpg"
    val result = lightPipeline.fullAnnotateImage(imagePath, "<REGION_TO_CATEGORY> region1")
    val answer = result("answer").head.asInstanceOf[Annotation]
    if (answer.metadata.contains("florence2_image")) {
      val florence2ImageBase64 = answer.metadata("florence2_image")
      val decodedFlorence2Image = java.util.Base64.getDecoder.decode(florence2ImageBase64)
      val fos = new FileOutputStream(
        new File(s"src/test/resources/images/florence2_image_${System.currentTimeMillis()}.png"))
      fos.write(decodedFlorence2Image)
      fos.close()
    }
    if (answer.metadata.contains("florence2_postprocessed_raw")) {
      val florence2PostprocessedRaw = answer.metadata("florence2_postprocessed_raw")
      println(s"florence2PostprocessedRaw: $florence2PostprocessedRaw")
    }
    println(s"<REGION_TO_CATEGORY> result: ${answer.result}")
    assert(answer.result.nonEmpty)
  }

  it should "run region to description (<REGION_TO_DESCRIPTION> region1)" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/car.jpg"
    val result = lightPipeline.fullAnnotateImage(imagePath, "<REGION_TO_DESCRIPTION> region1")
    val answer = result("answer").head.asInstanceOf[Annotation]
    if (answer.metadata.contains("florence2_image")) {
      val florence2ImageBase64 = answer.metadata("florence2_image")
      val decodedFlorence2Image = java.util.Base64.getDecoder.decode(florence2ImageBase64)
      val fos = new FileOutputStream(
        new File(s"src/test/resources/images/florence2_image_${System.currentTimeMillis()}.png"))
      fos.write(decodedFlorence2Image)
      fos.close()
    }
    if (answer.metadata.contains("florence2_postprocessed_raw")) {
      val florence2PostprocessedRaw = answer.metadata("florence2_postprocessed_raw")
      println(s"florence2PostprocessedRaw: $florence2PostprocessedRaw")
    }
    println(s"<REGION_TO_DESCRIPTION> result: ${answer.result}")
    assert(answer.result.nonEmpty)
  }

  it should "run region to OCR (<REGION_TO_OCR> region1)" taggedAs SlowTest in {
    val lightPipeline = new LightPipeline(model)
    val imagePath = "src/test/resources/images/ocr_sample.jpg"
    val result = lightPipeline.fullAnnotateImage(imagePath, "<REGION_TO_OCR> region1")
    val answer = result("answer").head.asInstanceOf[Annotation]
    if (answer.metadata.contains("florence2_image")) {
      val florence2ImageBase64 = answer.metadata("florence2_image")
      val decodedFlorence2Image = java.util.Base64.getDecoder.decode(florence2ImageBase64)
      val fos = new FileOutputStream(
        new File(s"src/test/resources/images/florence2_image_${System.currentTimeMillis()}.png"))
      fos.write(decodedFlorence2Image)
      fos.close()
    }
    if (answer.metadata.contains("florence2_postprocessed_raw")) {
      val florence2PostprocessedRaw = answer.metadata("florence2_postprocessed_raw")
      println(s"florence2PostprocessedRaw: $florence2PostprocessedRaw")
    }
    println(s"<REGION_TO_OCR> result: ${answer.result}")
    assert(answer.result.nonEmpty)
  }

  private def getFlorence2TransformerPipelineModel = {
    val testDF = getTestDF

    val imageAssembler: ImageAssembler = new ImageAssembler()
      .setInputCol("image")
      .setOutputCol("image_assembler")

    val loadModel = Florence2Transformer
      .loadSavedModel(
        "/mnt/research/Projects/ModelZoo/Florence2/models/int4/microsoft/Florence-2-base-ft",
        ResourceHelper.spark)
      .setInputCols("image_assembler")
      .setOutputCol("answer")
      .setMaxOutputLength(50)

    val newPipeline: Pipeline =
      new Pipeline().setStages(Array(imageAssembler, loadModel))

    newPipeline.fit(testDF)
  }

  private def getTestDF: DataFrame = {
    val imageFolder = "src/test/resources/handwriting/"
    val imageDF: DataFrame = ResourceHelper.spark.read
      .format("image")
      .option("dropInvalid", value = true)
      .load(imageFolder)

    val testDF: DataFrame = imageDF.withColumn("text", lit("<OCR_WITH_REGION>"))

    testDF
  }

}
