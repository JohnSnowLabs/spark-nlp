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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.util.{DefaultParamsReadable, MLReader}

trait HasPretrained[M <: PipelineStage] {

  /** Only MLReader types can use this interface */
  this: { def read: MLReader[M] } =>

  val defaultModelName: Option[String]

  val defaultLang: String = "en"

  val defaultPreferredEngine: String = "onnx"

  lazy val defaultLoc: String = ResourceDownloader.publicLoc

  implicit private val companion: DefaultParamsReadable[M] =
    this.asInstanceOf[DefaultParamsReadable[M]]

  private val errorMsg =
    s"${this.getClass.getName} does not have a default pretrained model. Please provide a model name."

  /** Java default argument interoperability */
  def pretrained(
      name: String,
      lang: String,
      remoteLoc: String,
      preferredEngine: String = "onnx"): M = {
    if (Option(name).isEmpty)
      throw new NotImplementedError(errorMsg)
    ResourceDownloader.downloadModel(companion, name, Option(lang), remoteLoc, preferredEngine)
  }

  def pretrained(name: String, lang: String): M =
    pretrained(name, lang, defaultLoc, defaultPreferredEngine)

  def pretrained(name: String): M =
    pretrained(name, defaultLang, defaultLoc, defaultPreferredEngine)

  def pretrained(): M =
    pretrained(
      defaultModelName.getOrElse(throw new Exception(errorMsg)),
      defaultLang,
      defaultLoc,
      defaultPreferredEngine)

  def pretrained(name: String, lang: String, remoteLoc: String): M =
    pretrained(name, lang, remoteLoc, defaultPreferredEngine)
  def pretrainedEngine(name: String, preferredEngine: String): M =
    pretrained(name, defaultLang, defaultLoc, preferredEngine)

  def pretrainedEngine(name: String, lang: String, preferredEngine: String): M =
    pretrained(name, lang, defaultLoc, preferredEngine)

}
