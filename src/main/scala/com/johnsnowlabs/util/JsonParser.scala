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
