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

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.JsonParser
import org.json4s.jackson.JsonMethods
import org.json4s.{JNothing, JValue}

import scala.io.Source

private[johnsnowlabs] case class Preprocessor(
    do_normalize: Boolean = true,
    do_resize: Boolean,
    feature_extractor_type: String,
    image_mean: Array[Double],
    image_std: Array[Double],
    resample: Int,
    size: Int)

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

  def schemaCheckViT(jsonStr: String): Boolean = {
    val json = JsonMethods.parse(jsonStr)
    val schemaCorrect =
      if (json.has("do_normalize") && json.has("do_resize") && json.has("image_mean") && json
          .has("image_std") && json.has("resample") && json.has("size")) true
      else false

    schemaCorrect
  }

  def loadPreprocessorConfig(preprocessorConfigPath: String): Preprocessor = {
    val preprocessorConfigStream =
      ResourceHelper.getResourceStream(preprocessorConfigPath)
    val preprocessorConfigJsonContent =
      Source.fromInputStream(preprocessorConfigStream).mkString

    val preprocessorJsonErrorMsg =
      s"""The schema of preprocessor_config.json file is incorrect. It should look like this:         
         |{
         |  "do_normalize": true,
         |  "do_resize": true,
         |  "feature_extractor_type": "ViTFeatureExtractor",
         |  "image_mean": [
         |    0.5,
         |    0.5,
         |    0.5
         |  ],
         |  "image_std": [
         |    0.5,
         |    0.5,
         |    0.5
         |  ],
         |  "resample": 2,
         |  "size": 224
         |}
         |""".stripMargin
    require(Preprocessor.schemaCheckViT(preprocessorConfigJsonContent), preprocessorJsonErrorMsg)

    val preprocessorConfig =
      try {
        JsonParser.parseObject[Preprocessor](preprocessorConfigJsonContent)
      } catch {
        case e: Exception =>
          println(s"${preprocessorJsonErrorMsg} \n error: ${e.getMessage}")
          JsonParser.parseObject[Preprocessor](preprocessorConfigJsonContent)
      }
    preprocessorConfig
  }
}
