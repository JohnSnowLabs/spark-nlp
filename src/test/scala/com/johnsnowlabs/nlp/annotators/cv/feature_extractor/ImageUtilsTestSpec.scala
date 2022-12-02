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
import org.scalatest.flatspec.AnyFlatSpec

import java.awt.color.ColorSpace
import java.awt.image.BufferedImage
import java.io.{File, InputStream}
import scala.io.Source

class ImageUtilsTestSpec extends AnyFlatSpec {

  val preprocessorConfigPath = new File(
    "src/test/resources/image_preprocessor/preprocessor_config.json")

  val preprocessorConfigStream: InputStream =
    ResourceHelper.getResourceStream(preprocessorConfigPath.getAbsolutePath)
  val preprocessorConfigJsonContent: String =
    Source.fromInputStream(preprocessorConfigStream).mkString
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

}
