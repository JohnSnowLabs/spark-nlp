/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.partition

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.spark.ml.param.Param
import scala.collection.JavaConverters._

trait HasHTMLReaderProperties extends ParamsAndFeaturesWritable {

  val timeout = new Param[Int](
    this,
    "timeout",
    "Timeout value in seconds for reading remote HTML resources. Applied when fetching content from URLs.")

  def setTimeout(value: Int): this.type = set(timeout, value)

  val headers =
    new Param[Map[String, String]](this, "headers", "HTTP headers to include in requests")

  def setHeaders(value: Map[String, String]): this.type = set(headers, value)

  def setHeadersPython(headers: java.util.Map[String, String]): this.type = {
    setHeaders(headers.asScala.toMap)
  }

  val includeTitleTag = new Param[Boolean](
    this,
    "includeTitleTag",
    "Whether to include the title tag in the HTML output. Default is false.")

  def setIncludeTitleTag(value: Boolean): this.type = set(includeTitleTag, value)

  val outputFormat = new Param[String](
    this,
    "outputFormat",
    "Output format for the table content. Options are 'plain-text' or 'html-table'. Default is 'json-table'.")

  def setOutputFormat(value: String): this.type = set(outputFormat, value)

  setDefault(timeout -> 0, includeTitleTag -> false, headers -> Map.empty[String, String])

}
