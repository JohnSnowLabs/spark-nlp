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
package com.johnsnowlabs.reader.util

import scala.util.Try

object PartitionOptions {

  def getDefaultBoolean(
      params: Map[String, String],
      options: Seq[String],
      default: Boolean): Boolean = {
    options
      .flatMap(params.get)
      .map(_.trim.toLowerCase)
      .flatMap(value => Try(value.toBoolean).toOption)
      .headOption
      .getOrElse(default)
  }

  def getDefaultInt(params: Map[String, String], options: Seq[String], default: Int): Int = {
    options
      .flatMap(params.get)
      .flatMap(value => Try(value.toInt).toOption)
      .headOption
      .getOrElse(default)
  }

  def getDefaultString(
      params: Map[String, String],
      options: Seq[String],
      default: String): String = {
    options
      .flatMap(params.get)
      .flatMap(value => Try(value).toOption)
      .headOption
      .getOrElse(default)
  }

}
