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

package com.johnsnowlabs.nlp.annotators.cv.feature_extractor

import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.ImageResizeUtils
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.TestUtils.{assertPixels, readFile}
import com.johnsnowlabs.util.{Benchmark, JsonParser}
import org.apache.spark.ml.image.ImageSchema
import org.json4s.JsonAST.{JInt, JObject, JString}
import org.json4s.jackson.JsonMethods._
import org.scalatest.flatspec.AnyFlatSpec

import java.awt.Color
import java.awt.color.ColorSpace
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

class ImageUtilsTestSpec extends AnyFlatSpec {

  val preprocessorConfigPath =
    "src/test/resources/image_preprocessor/preprocessor_config.json"

  val preprocessorConfigJsonContent: String = readFile(preprocessorConfigPath)

  val preprocessorConfig: Preprocessor =
    JsonParser.parseObject[Preprocessor](preprocessorConfigJsonContent)

  val imageBufferedImage: BufferedImage =
    ImageIOUtils.loadImage("src/test/resources/image/egyptian_cat.jpeg").get
  val isGray: Boolean =
    imageBufferedImage.getColorModel.getColorSpace.getType == ColorSpace.TYPE_GRAY
  val hasAlpha: Boolean = imageBufferedImage.getColorModel.hasAlpha

  val (nChannels, mode) = if (isGray) {
    (1, ImageSchema.ocvTypes("CV_8UC1"))
  } else if (hasAlpha) {
    (4, ImageSchema.ocvTypes("CV_8UC4"))
  } else {
    (3, ImageSchema.ocvTypes("CV_8UC3"))
  }

  val resizedImage: BufferedImage =
    ImageResizeUtils.resizeBufferedImage(
      width = preprocessorConfig.size,
      height = preprocessorConfig.size,
      resample = preprocessorConfig.resample)(imageBufferedImage)

  "ImageResizeUtils" should "resize and normalize an image" taggedAs FastTest in {

    Benchmark.measure(iterations = 1, forcePrint = true, description = "Time to load image") {
      ImageIOUtils.loadImage("src/test/resources/image/egyptian_cat.jpeg")
    }

    Benchmark.measure(
      iterations = 1000,
      forcePrint = true,
      description = "Time to resizeBufferedImage an image") {
      ImageResizeUtils.resizeBufferedImage(
        width = preprocessorConfig.size,
        height = preprocessorConfig.size,
        resample = preprocessorConfig.resample)(imageBufferedImage)
    }

    Benchmark.measure(
      iterations = 1,
      forcePrint = true,
      description = "Time to normalize the resized image") {
      ImageResizeUtils.normalizeAndConvertBufferedImage(
        resizedImage,
        preprocessorConfig.image_mean,
        preprocessorConfig.image_std,
        preprocessorConfig.do_normalize,
        preprocessorConfig.do_rescale,
        preprocessorConfig.rescale_factor)
    }

    Benchmark.measure(
      iterations = 1,
      forcePrint = true,
      description = "Time to normalize with 0.0d") {
      ImageResizeUtils.normalizeAndConvertBufferedImage(
        resizedImage,
        Array(0.0d, 0.0d, 0.0d),
        Array(0.0d, 0.0d, 0.0d),
        preprocessorConfig.do_normalize,
        preprocessorConfig.do_rescale,
        preprocessorConfig.rescale_factor)
    }

  }

  "ImageResizeUtils" should "read preprocessor_config.json file" taggedAs FastTest in {
    val preprocessorConfig =
      Preprocessor.loadPreprocessorConfig(preprocessorConfigJsonContent)

    assert(preprocessorConfig.feature_extractor_type == "ViTFeatureExtractor")
    assert(preprocessorConfig.image_mean sameElements Array(0.5d, 0.5d, 0.5d))
  }

  // Some models don't have feature_extractor_type in the config json
  "ImageResizeUtils" should "rename image processor if feature_extractor_type not available" taggedAs FastTest in {
    val json = parse(preprocessorConfigJsonContent) transformField {
      case ("feature_extractor_type", _) =>
        ("image_processor_type", JString("ViTImageProcessor"))
    }

    val contents: String = compact(render(json))

    val preprocessorConfig =
      Preprocessor.loadPreprocessorConfig(contents)

    assert(preprocessorConfig.feature_extractor_type == "ViTFeatureExtractor")
  }

  "ImageResizeUtils" should "throw exceptions if config is malformed" taggedAs FastTest in {

    val differingWHJson = parse(preprocessorConfigJsonContent) transformField {
      case ("size", JInt(num)) =>
        val sizes = JObject(List(("height", JInt(num)), ("width", JInt(num * 2))))
        ("size", sizes)
    }
    val differingWHContents: String = compact(render(differingWHJson))

    assertThrows[IllegalArgumentException] {
      Preprocessor.loadPreprocessorConfig(differingWHContents)
    }

    val jsonNoName = parse(preprocessorConfigJsonContent) removeField {
      case ("feature_extractor_type", _) | ("image_processor_type", _) => true
      case _ => false
    }

    val contentsNoName: String = compact(render(jsonNoName))

    assertThrows[IllegalArgumentException] {
      Preprocessor.loadPreprocessorConfig(contentsNoName)
    }

    val jsonPath = "src/test/resources/image_preprocessor/preprocessor_config_swin.json"
    val swinContent = readFile(jsonPath)
    // height and width not identical
    val swinJson = parse(swinContent)

    val noRescale = swinJson removeField {
      case ("rescale_factor", _) => true
      case _ => false
    }

    val contentsNoRescale: String = compact(render(noRescale))

    assertThrows[IllegalArgumentException] {
      Preprocessor.loadPreprocessorConfig(contentsNoRescale)
    }

  }

  "ImageResizeUtils" should "normalize an image correctly with custom rescale_factor" taggedAs FastTest in {
    val jsonPath = "src/test/resources/image_preprocessor/preprocessor_config_swin.json"
    val preprocessorConfig =
      Preprocessor.loadPreprocessorConfig(readFile(jsonPath))

    val normalized = ImageResizeUtils
      .normalizeAndConvertBufferedImage(
        resizedImage,
        preprocessorConfig.image_mean,
        preprocessorConfig.image_std,
        preprocessorConfig.do_normalize,
        preprocessorConfig.do_rescale,
        preprocessorConfig.rescale_factor)

    def normalize(color: Int, mean: Double, std: Double): Float = {
      (((color * preprocessorConfig.rescale_factor) - mean) / std).toFloat
    }

    (0 until resizedImage.getWidth).zip(0 until resizedImage.getHeight).map { case (x, y) =>
      val originalColor = new Color(resizedImage.getRGB(x, y))
      val red = normalized(0)

      assert(
        normalize(
          originalColor.getRed,
          preprocessorConfig.image_mean(0),
          preprocessorConfig.image_std(0)) == red(x)(y))
    }

  }

  "ImageResizeUtils" should "resize and crop image" taggedAs FastTest in {
    val preprocessorJsonPath =
      "src/test/resources/image_preprocessor/preprocessor_config_convnext.json"

    val preprocessor =
      Preprocessor.loadPreprocessorConfig(readFile(preprocessorJsonPath))

    val img = ImageIOUtils.loadImage("src/test/resources/image/ox.JPEG").get

    val processedImage =
      ImageResizeUtils.resizeAndCenterCropImage(
        img,
        preprocessor.size,
        preprocessor.resample,
        preprocessor.crop_pct.get)

    assert(processedImage.getWidth == 224)
    assert(processedImage.getHeight == 224)

    // Use PNG so no compression is applied
    val expectedCropped =
      ImageIO.read(new File("src/test/resources/image_preprocessor/ox_cropped.png"))

    (0 until processedImage.getWidth).zip(0 until processedImage.getHeight).map { case (x, y) =>
      assert(
        expectedCropped.getRGB(x, y) == processedImage.getRGB(x, y),
        s"Pixel did not match for coordinates ($x, $y)")
    }

    // Case: Image is too small for size
    val smallImg =
      ImageIOUtils.loadImage("src/test/resources/image_preprocessor/ox_small.JPEG").get

    val processedSmallImage =
      ImageResizeUtils.resizeAndCenterCropImage(
        smallImg,
        preprocessor.size,
        preprocessor.resample,
        preprocessor.crop_pct.get)

    assert(processedSmallImage.getWidth == 224)
    assert(processedSmallImage.getHeight == 224)
  }

}
