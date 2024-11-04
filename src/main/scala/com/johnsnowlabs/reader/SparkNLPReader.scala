/*
 * Copyright 2017-2024 John Snow Labs
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
package com.johnsnowlabs.reader

import org.apache.spark.sql.DataFrame

import java.util
import scala.collection.JavaConverters._

class SparkNLPReader(params: java.util.Map[String, String] = new util.HashMap()) {

  def html(htmlPath: String): DataFrame = {
    val titleFontSize = params.asScala.getOrElse("titleFontSize", "16")
    val htmlReader = new HTMLReader(titleFontSize.toInt)
    htmlReader.read(htmlPath)
  }

  def html(urls: Array[String]): DataFrame = {
    val titleFontSize = params.asScala.getOrElse("titleFontSize", "16")
    val htmlReader = new HTMLReader(titleFontSize.toInt)
    htmlReader.read(urls)
  }

}
