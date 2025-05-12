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

import com.johnsnowlabs.util.JsonParser
import org.json4s.{JNothing, JValue}

/** Case class represting an image pre-processor. Instances should be initialized with
  * loadPreprocessorConfig.
  *
  * @param do_normalize
  *   Whether to normalize the image by subtracting `mean` and dividing by `std`
  * @param do_resize
  *   Whether to resize the image to `size`
  * @param feature_extractor_type
  *   Name of the feature extractor
  * @param image_mean
  *   Array of means, one value for each color channel
  * @param image_std
  *   Array of standard deviations, one value for each color channel
  * @param resample
  *   Integer representing an image filter to be used for resizing/resampling. Corresponds to
  *   constants defined in PIL (either PIL.Image.NEAREST, PIL.Image.BILINEAR or PIL.Image.BICUBIC
  *   supported).
  * @param size
  *   Size of the image after processing
  * @param do_rescale
  *   Whether to rescale color values to rescale_factor
  * @param rescale_factor
  *   Factor to rescale color values by
  * @param crop_pct
  *   Percentage to crop the image. If set, first scales the image, then crops it to arrive at
  *   `size`
  */
private[johnsnowlabs] case class Preprocessor(
    do_normalize: Boolean = true,
    do_resize: Boolean = true,
    feature_extractor_type: String,
    image_mean: Array[Double],
    image_std: Array[Double],
    resample: Int,
    size: Int,
    do_rescale: Boolean = true,
    rescale_factor: Double = 1 / 255.0d,
    crop_pct: Option[Double] = None)

private[johnsnowlabs] case class PreprocessorConfig(
    do_normalize: Boolean,
    do_resize: Boolean,
    feature_extractor_type: Option[String],
    image_processor_type: Option[String],
    image_mean: Array[Double],
    image_std: Array[Double],
    resample: Int,
    size: Any,
    do_rescale: Option[Boolean],
    rescale_factor: Option[Double],
    crop_pct: Option[Double])

private[johnsnowlabs] object Preprocessor {
  def apply(
      do_normalize: Boolean,
      do_resize: Boolean,
      feature_extractor_type: String,
      image_mean: Array[Double],
      image_std: Array[Double],
      resample: Int,
      size: Int): Preprocessor = {

    // if more complex manipulation is required
    new Preprocessor(
      do_normalize,
      do_resize,
      feature_extractor_type,
      image_mean,
      image_std,
      resample,
      size)
  }

  private implicit class JValueExtended(value: JValue) {
    def has(childString: String): Boolean = {
      (value \ childString) != JNothing
    }
  }

  /** Loads in initializes a Preprocessor from a json string.
    *
    * @param preprocessorConfigJsonContent
    *   Json contents in a String
    * @return
    *   Loaded Preprocessor
    */
  def loadPreprocessorConfig(preprocessorConfigJsonContent: String): Preprocessor = {

    val preprocessorJsonErrorMsg =
      s"""The schema of preprocessor_config.json file is incorrect. It should look like this:
         |{
         |  "do_normalize": bool,
         |  "do_resize": bool,
         |  ("feature_extractor_type"|"image_processor_type"): string,
         |  "image_mean": Array[double],
         |  "image_std": Array[double,
         |  "resample": int,
         |  "size": int,
         |  ["do_rescale": bool],
         |  ["rescale_factor": double],
         |  ["crop_pct": double]
         |}
         |""".stripMargin

    def parseSize(config: PreprocessorConfig) = {
      config.size match {
        case sizeMap: Map[String, BigInt] if sizeMap.contains("width") =>
          val width = sizeMap("width")
          require(
            width == sizeMap("height"),
            "Different sizes for width and height are currently not supported.")
          width.toInt
        case sizeMap: Map[String, BigInt] if sizeMap.contains("shortest_edge") =>
          // ConvNext case: Size of the output image after `resize` has been applied
          sizeMap("shortest_edge").toInt
        case sizeMap: Map[String, BigInt] if sizeMap.contains("longest_edge") =>
          // ConvNext case: Size of the output image after `resize` has been applied
          sizeMap("longest_edge").toInt
        case sizeInt: BigInt => sizeInt.toInt
        case sizeMap: Map[String, BigInt] if sizeMap.contains("max_pixels") =>
          val max_pixels = sizeMap("max_pixels")
          max_pixels.toInt
        case _ =>
          throw new IllegalArgumentException(
            "Unsupported format for size. Should either be int or dict with entries \'width\' and \'height\' or \'shortest_edge\'")
      }
    }

    def parseExtractorType(config: PreprocessorConfig) = {
      config.feature_extractor_type.getOrElse({
        config.image_processor_type
          .getOrElse(throw new IllegalArgumentException(
            "Either \'feature_extractor_type\' or \'image_processor_type\' should be set."))
          .replace("ImageProcessor", "FeatureExtractor")
      })
    }

    def parseRescaleFactor(config: PreprocessorConfig) = {
      if (config.do_rescale.isDefined) config.rescale_factor.getOrElse {
        throw new IllegalArgumentException(
          "Value do_rescale is true but no rescale_factor found in config.")
      }
      else 1 / 255.0d
    }

    val preprocessorConfig =
      try {
        val config = JsonParser.parseObject[PreprocessorConfig](preprocessorConfigJsonContent)

        // json4s parses BigInt by default
        val size: Int = parseSize(config)

        val extractorType = parseExtractorType(config)

        val rescaleFactor: Double =
          parseRescaleFactor(config) // Default value

        val doRescale = config.do_rescale.getOrElse(true)

        Preprocessor(
          do_normalize = config.do_normalize,
          do_resize = config.do_resize,
          feature_extractor_type = extractorType,
          image_mean = config.image_mean,
          image_std = config.image_std,
          resample = config.resample,
          size = size,
          do_rescale = doRescale,
          rescale_factor = rescaleFactor,
          crop_pct = config.crop_pct)
      } catch {
        case e: Exception =>
          println(s"$preprocessorJsonErrorMsg \n error: ${e.getMessage}")
          throw e
      }

    preprocessorConfig
  }
}
