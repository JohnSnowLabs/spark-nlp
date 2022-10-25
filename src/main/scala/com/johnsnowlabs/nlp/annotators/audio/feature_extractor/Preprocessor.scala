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

package com.johnsnowlabs.nlp.annotators.audio.feature_extractor

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.JsonParser
import org.json4s.jackson.JsonMethods
import org.json4s.{JNothing, JValue}

import scala.io.Source

private[johnsnowlabs] case class Preprocessor(
    do_normalize: Boolean = true,
    feature_size: Int,
    padding_side: String,
    padding_value: Float,
    return_attention_mask: Boolean,
    sampling_rate: Int)

private[johnsnowlabs] object Preprocessor {
  def apply(
      do_normalize: Boolean = true,
      feature_size: Int,
      padding_side: String,
      padding_value: Float,
      return_attention_mask: Boolean,
      sampling_rate: Int): Preprocessor = {

    // if more complex manipulation is required
    new Preprocessor(
      do_normalize,
      feature_size,
      padding_side,
      padding_value,
      return_attention_mask,
      sampling_rate)
  }

  private implicit class JValueExtended(value: JValue) {
    def has(childString: String): Boolean = {
      (value \ childString) != JNothing
    }
  }

  def schemaCheckWav2Vec2(jsonStr: String): Boolean = {
    val json = JsonMethods.parse(jsonStr)
    val schemaCorrect =
      if (json.has("do_normalize") && json.has("feature_size") && json.has("padding_side") && json
          .has("padding_value") && json.has("return_attention_mask") && json.has("sampling_rate"))
        true
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
         |  "feature_size": 1,
         |  "padding_side": "right",
         |  "padding_value": 0.0,
         |  "return_attention_mask": false,
         |  "sampling_rate": 16000
         |}
         |""".stripMargin
    require(
      Preprocessor.schemaCheckWav2Vec2(preprocessorConfigJsonContent),
      preprocessorJsonErrorMsg)

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
