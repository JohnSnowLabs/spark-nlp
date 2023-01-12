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
import org.json4s.jackson.JsonMethods
import org.json4s.{JNothing, JValue}

private[johnsnowlabs] case class Preprocessor(
    do_normalize: Boolean = true,
    do_resize: Boolean = true,
    feature_extractor_type: String,
    image_mean: Array[Double],
    image_std: Array[Double],
    resample: Int,
    size: Int,
    do_rescale: Boolean = true,
    rescale_factor: Double = 1 / 255.0d)

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
    rescale_factor: Option[Double])

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
         |  ["rescale_factor": double]
         |}
         |""".stripMargin

    val preprocessorConfig =
      try {
        val config = JsonParser.parseObject[PreprocessorConfig](preprocessorConfigJsonContent)

        // json4s parses BigInt by default
        val size: Int = config.size match {
          case sizeMap: Map[String, BigInt] =>
            val width = sizeMap("width")
            require(
              width == sizeMap("height"),
              "Different sizes for width and height are currently not supported.")
            width.toInt
          case sizeInt: BigInt => sizeInt.toInt
          case _ =>
            throw new IllegalArgumentException(
              "Unsupported format for size. Should either be int or dict with entries \'width\' and \'height\'")
        }

        val extractorType = config.feature_extractor_type.getOrElse({
          config.image_processor_type
            .getOrElse(throw new IllegalArgumentException(
              "Either \'feature_extractor_type\' or \'image_processor_type\' should be set."))
            .replace("ImageProcessor", "FeatureExtractor")
        })

        val doRescale = config.do_rescale.getOrElse(false)

        val rescaleFactor: Double = if (doRescale) config.rescale_factor.getOrElse {
          throw new IllegalArgumentException(
            "Value do_rescale is true but no rescale_factor found in config.")
        }
        else 1 / 255.0d // Default value

        Preprocessor(
          do_normalize = config.do_normalize,
          do_resize = config.do_resize,
          extractorType,
          config.image_mean,
          config.image_std,
          config.resample,
          size,
          doRescale,
          rescaleFactor)
      } catch {
        case e: Exception =>
          println(s"$preprocessorJsonErrorMsg \n error: ${e.getMessage}")
          throw e
      }

    preprocessorConfig
  }
}
