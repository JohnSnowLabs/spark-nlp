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

import io.circe.{Decoder, Error, parser}

class JsonParser[A] {

  def readJson(jsonContent: String)(implicit decoder: Decoder[A]): A = {
   val decodedObject: Either[Error, A] = parser.decode[A](jsonContent)
    decodedObject match {
      case Left(error) => throw new UnsupportedOperationException(s"Error reading JSON: ${error.getMessage}")
      case Right(decoded) => decoded
    }
  }

  def readJsonArray(jsonContent: String)(implicit decoder: Decoder[Array[A]]): Array[A] = {
    val decodedObjects: Either[Error, Array[A]] = parser.decode[Array[A]](jsonContent)
    decodedObjects match {
      case Left(error) => throw new UnsupportedOperationException(s"Error reading JSON as array: ${error.getMessage}")
      case Right(decodedArray) => decodedArray
    }
  }

}
