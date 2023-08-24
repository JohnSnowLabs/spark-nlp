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

import com.johnsnowlabs.util.JsonParser
import org.json4s.jackson.JsonMethods
import org.json4s.{Formats, JNothing, JValue}

import scala.collection.mutable.ArrayBuffer
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
    def hasAttributes(attributes: Seq[String]): Boolean =
      attributes.forall(value.has(_))
  }

  def checkSchema(json: JValue, processorClass: String): Unit = {
    val attributes = processorClass match {
      case "Wav2Vec2Processor" =>
        PreprocessorAttributes.Wave2Vec
      case "WhisperProcessor" =>
        PreprocessorAttributes.Whisper
      case other => throw new IllegalArgumentException(s"Preprocessor for $other not supported.")
    }

    val schemaCorrect = json.hasAttributes(attributes)

    require(
      schemaCorrect,
      s"The schema of preprocessor_config.json file is incorrect. It should have the following fields:\n${attributes
          .mkString("\n")}")
  }

  def loadPreprocessorConfig(preprocessorConfigJsonContent: String): Preprocessor = {
    implicit val formats: Formats = org.json4s.DefaultFormats

    val parsedJson = JsonMethods.parse(preprocessorConfigJsonContent)

    val processorClass = (parsedJson \ "processor_class").extractOrElse[String](
      throw new Exception("\"processor_class\" attribute not found in preprocessor_config.json!"))

    Preprocessor.checkSchema(parsedJson, processorClass)

    val preprocessorConfig: Preprocessor =
      try {
        processorClass match {
          case "WhisperProcessor" =>
            JsonParser.parseObject[WhisperPreprocessor](preprocessorConfigJsonContent)
          case _ => JsonParser.parseObject[Preprocessor](preprocessorConfigJsonContent)
        }
      } catch {
        case e: Exception =>
          println(s"Could not parse preprocessor config ${e.getClass.toString}:${e.getMessage}")
          throw e
      }
    preprocessorConfig
  }

  def normalize(rawAudio: Array[Float]): Array[Float] = {
    val normalizedData = ArrayBuffer[Float]()
    val meanData = mean(rawAudio)
    val varianceData = variance(rawAudio)
    for (x <- rawAudio) {
      normalizedData += (x - meanData) / scala.math.sqrt(varianceData + 0.0000001).toFloat
    }
    normalizedData.toArray
  }

  def mean(rawAudio: Array[Float]): Float =
    if (rawAudio.isEmpty) 0 else rawAudio.sum / rawAudio.length

  def variance(rawAudio: Array[Float]): Double = {
    val avg = mean(rawAudio)
    rawAudio.map(a => math.pow(a - avg, 2)).sum / rawAudio.length
  }

  // TODO: Check for infinite recursion
  def removeDup(lst: List[Int]): List[Int] = {
    lst match {
      case head :: tail =>
        val (duplicateList, remainList) = lst.span(_ == head)
        duplicateList.head :: removeDup(remainList)
      case Nil => List()
    }
  }

  def pad(
      audio: Array[Float],
      paddingValue: Float,
      totalLength: Int,
      paddingSide: String): Array[Float] = {
    val padding = Array.fill(totalLength - audio.length)(paddingValue)

    paddingSide match {
      case "left" => padding ++ audio
      case "right" => audio ++ padding
      case _ => audio ++ padding
    }
  }

  def truncate(audio: Array[Float], maxLength: Int): Array[Float] =
    if (audio.length > maxLength) audio.slice(0, maxLength) else audio

}
