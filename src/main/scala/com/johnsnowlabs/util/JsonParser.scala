/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.util

import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, Formats}

object JsonParser {

  implicit var formats: Formats = DefaultFormats

  def parseObject[A](json: String)(implicit manifest: Manifest[A]): A = {
    val parsed = JsonMethods.parse(json)
    parsed.extract[A]
  }

  def parseArray[A](json: String)(implicit manifest: Manifest[A]): Array[A] = {
    val parsed = JsonMethods.parse(json)
    parsed.extract[Array[A]]
  }

}