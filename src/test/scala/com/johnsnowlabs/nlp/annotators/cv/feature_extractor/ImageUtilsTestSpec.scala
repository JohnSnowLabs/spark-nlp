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
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.{Benchmark, JsonParser}
import org.apache.spark.ml.image.ImageSchema
import org.json4s.JsonAST.{JInt, JObject, JString}
import org.json4s.jackson.JsonMethods._
import org.scalatest.flatspec.AnyFlatSpec

import java.awt.color.ColorSpace
import java.awt.image.BufferedImage
import java.io.File
import scala.io.Source

class ImageUtilsTestSpec extends AnyFlatSpec {

  def readJson(path: String): String = {
    val stream = ResourceHelper.getResourceStream(new File(path).getAbsolutePath)
    Source.fromInputStream(stream).mkString
  }

  val preprocessorConfigPath =
    "src/test/resources/image_preprocessor/preprocessor_config.json"

  val preprocessorConfigJsonContent: String = readJson(preprocessorConfigPath)

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
      Some(nChannels))(imageBufferedImage)

  "ImageResizeUtils" should "resize and normalize an image" taggedAs FastTest in {

    Benchmark.measure(iterations = 10, forcePrint = true, description = "Time to load image") {
      ImageIOUtils.loadImage("src/test/resources/image/egyptian_cat.jpeg")
    }

    Benchmark.measure(
      iterations = 10,
      forcePrint = true,
      description = "Time to resizeBufferedImage an image") {
      ImageResizeUtils.resizeBufferedImage(
        width = preprocessorConfig.size,
        height = preprocessorConfig.size,
        Some(nChannels))(imageBufferedImage)
    }

    Benchmark.measure(
      iterations = 10,
      forcePrint = true,
      description = "Time to normalize the resized image") {
      ImageResizeUtils.normalizeBufferedImage(
        resizedImage,
        preprocessorConfig.image_mean,
        preprocessorConfig.image_std)
    }

    Benchmark.measure(
      iterations = 10,
      forcePrint = true,
      description = "Time to normalize with 0.0d") {
      ImageResizeUtils.normalizeBufferedImage(
        resizedImage,
        Array(0.0d, 0.0d, 0.0d),
        Array(0.0d, 0.0d, 0.0d))
    }

  }

  "ImageResizeUtils" should "read preprocessor_config.json file" taggedAs FastTest in {
    val preprocessorConfig =
      Preprocessor.loadPreprocessorConfig(preprocessorConfigJsonContent)

    assert(preprocessorConfig.feature_extractor_type == "ViTFeatureExtractor")
    assert(preprocessorConfig.image_mean sameElements Array(0.5d, 0.5d, 0.5d))
  }

  "ImageResizeUtils" should "read swin preprocessor_config.json file" taggedAs FastTest in {
    val jsonPath = "src/test/resources/image_preprocessor/preprocessor_config_swin.json"
    val preprocessorConfig =
      Preprocessor.loadPreprocessorConfig(readJson(jsonPath))

    assert(preprocessorConfig.do_normalize)
    assert(preprocessorConfig.do_rescale)
    assert(preprocessorConfig.do_resize)
    assert(preprocessorConfig.feature_extractor_type == "ViTFeatureExtractor")
    assert(preprocessorConfig.image_mean sameElements Array(0.485d, 0.456d, 0.406d))
    assert(preprocessorConfig.image_std sameElements Array(0.229d, 0.224d, 0.225d))
    assert(preprocessorConfig.resample == 3)
    assert(preprocessorConfig.rescale_factor == 0.00392156862745098)
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
    val swinContent = readJson(jsonPath)
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
      Preprocessor.loadPreprocessorConfig(readJson(jsonPath))

    val normalized = ImageResizeUtils
      .normalizeBufferedImage(
        resizedImage,
        preprocessorConfig.image_mean,
        preprocessorConfig.image_std,
        preprocessorConfig.rescale_factor)

    val expectedValues =
      JsonParser.parseArray[Array[Array[Float]]](
        readJson("src/test/resources/image_preprocessor/normalized_egyptian_cat.json"))

    val channels = normalized.length
    val width = normalized.head.length
    val height = normalized.head.head.length

    (0 until channels).foreach { channel =>
      (0 until width).foreach { w =>
        (0 until height).foreach { h =>
          assert(normalized(channel)(w)(h) == expectedValues(channel)(w)(h))
        }
      }
    }
  }

}
