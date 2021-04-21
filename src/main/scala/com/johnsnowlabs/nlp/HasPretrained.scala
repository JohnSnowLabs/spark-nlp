/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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

  lazy val defaultLoc: String = ResourceDownloader.publicLoc

  implicit private val companion: DefaultParamsReadable[M] = this.asInstanceOf[DefaultParamsReadable[M]]

  private val errorMsg = s"${this.getClass.getName} does not have a default pretrained model. Please provide a model name."

  /** Java default argument interoperability */

  def pretrained(name: String, lang: String, remoteLoc: String): M = {
    if (Option(name).isEmpty)
      throw new NotImplementedError(errorMsg)
    ResourceDownloader.downloadModel(companion, name, Option(lang), remoteLoc)
  }

  def pretrained(name: String, lang: String): M = pretrained(name, lang, defaultLoc)

  def pretrained(name: String): M = pretrained(name, defaultLang, defaultLoc)

  def pretrained(): M = pretrained(defaultModelName.getOrElse(throw new Exception(errorMsg)), defaultLang, defaultLoc)

}