package com.johnsnowlabs.nlp.pretrained

import com.johnsnowlabs.tags.SlowTest
import org.scalatest.flatspec.AnyFlatSpec

class PretrainedPipelineTest extends AnyFlatSpec {

  "PretrainedPipeline" should "infer for text input" taggedAs SlowTest in {
    val pipeline = PretrainedPipeline("clean_slang")
    val slangText = "yo, what is wrong with ya?"

    val annotationResult = pipeline.fullAnnotate(slangText)

    assert(annotationResult.nonEmpty)

    val slangTexts = Array(slangText, slangText)

    val annotationsResults = pipeline.fullAnnotate(slangTexts)
    assert(annotationsResults.nonEmpty)
  }

  it should "infer for image input" taggedAs SlowTest in {
    val pipeline = PretrainedPipeline("pipeline_image_classifier_vit_dogs")
    val imagesPath = "src/test/resources/image/"

    val image = imagesPath + "chihuahua.jpg"
    val annotationResult = pipeline.fullAnnotate(image)

    assert(annotationResult.nonEmpty)

    val images = Array(image, image)
    val annotationsResults = pipeline.fullAnnotate(images)

    assert(annotationsResults.nonEmpty)

  }

  it should "infer with fullAnnotateImage" taggedAs SlowTest in {
    val pipeline = PretrainedPipeline("pipeline_image_classifier_vit_dogs")
    val imagesPath = "src/test/resources/image/"

    val image = imagesPath + "chihuahua.jpg"
    val annotationResult = pipeline.fullAnnotateImage(image)

    assert(annotationResult.nonEmpty)

    val images = Array(image, image)
    val annotationsResults = pipeline.fullAnnotateImage(images)

    assert(annotationsResults.nonEmpty)

  }

  it should " infer for audio input" taggedAs SlowTest in {
    val pathToFileWithFloats = "src/test/resources/audio/csv/audio_floats.csv"
    val bufferedSource = scala.io.Source.fromFile(pathToFileWithFloats)
    val rawFloats = bufferedSource
      .getLines()
      .map(_.split(",").head.trim.toFloat)
      .toArray
    bufferedSource.close

    val pipeline = PretrainedPipeline("pipeline_asr_wav2vec2_bilal_2022")
    val annotationResult = pipeline.fullAnnotate(rawFloats)

    assert(annotationResult.nonEmpty)

    val audios = Array(rawFloats, rawFloats)
    val annotationsResults = pipeline.fullAnnotate(rawFloats)

    assert(annotationsResults.nonEmpty)

  }

}
