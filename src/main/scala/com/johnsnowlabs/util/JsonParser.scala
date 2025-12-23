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

package com.johnsnowlabs.util

import org.json4s.JsonAST.{JArray, JBool, JInt, JString}
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, Formats, JValue}

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

  def asString(j: JValue): Option[String] = j match {
    case JString(s) => Some(s)
    case JBool(b) => Some(b.toString)
    case JInt(i) => Some(i.toString)
    case JArray(arr) => Some(arr.collect { case JString(s) => s }.mkString(","))
    case _ => None
  }

  def asStringArray(j: JValue): Option[Array[String]] = j match {
    case JArray(arr) => Some(arr.collect { case JString(s) => s }.toArray)
    case JString(s) => Some(Array(s))
    case _ => None
  }

  def asInt(j: JValue): Option[Int] = j match {
    case JInt(i) => Some(i.toInt)
    case JString(s) => scala.util.Try(s.toInt).toOption
    case _ => None
  }

  def asBoolean(j: JValue): Option[Boolean] = j match {
    case JBool(b) => Some(b)
    case JString(s) => scala.util.Try(s.toBoolean).toOption
    case _ => None
  }

}
