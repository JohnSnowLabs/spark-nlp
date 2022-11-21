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

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import scala.collection.Map

/** Represents [[ImageAssembler]]'s output parts and their details
  *
  * @param annotatorType
  *   Image annotator type
  * @param origin
  *   The origin of the image
  * @param height
  *   Height of the image in pixels
  * @param width
  *   Width of the image in pixels
  * @param nChannels
  *   Number of image channels
  * @param mode
  *   OpenCV-compatible type
  * @param result
  *   Result of the annotation
  * @param metadata
  *   Metadata of the annotation
  */
case class AnnotationImage(
    annotatorType: String,
    origin: String,
    height: Int,
    width: Int,
    nChannels: Int,
    mode: Int,
    result: Array[Byte],
    metadata: Map[String, String])
    extends IAnnotation {

  override def equals(obj: Any): Boolean = {
    obj match {
      case annotation: AnnotationImage =>
        this.annotatorType == annotation.annotatorType &&
        this.origin == annotation.origin &&
        this.height == annotation.height &&
        this.width == annotation.width &&
        this.nChannels == annotation.nChannels &&
        this.mode == annotation.mode &&
        this.result.sameElements(annotation.result) &&
        this.metadata == annotation.metadata
      case _ => false
    }
  }

  def getAnnotatorType: String = {
    annotatorType
  }

  def getOrigin: String = {
    origin
  }

  def getHeight: Int = {
    height
  }

  def getWidth: Int = {
    width
  }

  def getChannels: Int = {
    nChannels
  }

  def getMode: Int = {
    mode
  }

  def getMetadata: Map[String, String] = {
    metadata
  }

}

object AnnotationImage {

  case class AnnotationContainer(__annotation: Array[AnnotationImage])

  /** This is spark type of an annotation representing its metadata shape */
  val dataType = new StructType(
    Array(
      StructField("annotatorType", StringType, nullable = true),
      StructField("origin", StringType, nullable = false),
      StructField("height", IntegerType, nullable = false),
      StructField("width", IntegerType, nullable = false),
      StructField("nChannels", IntegerType, nullable = false),
      // OpenCV-compatible type: CV_8UC3 in most cases
      StructField("mode", IntegerType, nullable = false),
      // Bytes in OpenCV-compatible order: row-wise BGR in most cases
      StructField("result", BinaryType, nullable = false),
      StructField("metadata", MapType(StringType, StringType), nullable = true)))

  val arrayType = new ArrayType(dataType, true)

  case class ImageFields(
      origin: String,
      height: Int,
      width: Int,
      nChannels: Int,
      mode: Int,
      result: Array[Byte])

  /** This method converts a [[org.apache.spark.sql.Row]] into an [[AnnotationImage]]
    *
    * @param row
    *   spark row to be converted
    * @return
    *   AnnotationImage
    */
  def apply(row: Row): AnnotationImage = {
    AnnotationImage(
      row.getString(0),
      row.getString(1),
      row.getInt(2),
      row.getInt(3),
      row.getInt(4),
      row.getInt(5),
      row.getSeq[Byte](6).toArray,
      row.getMap[String, String](7))
  }

  def apply(image: ImageFields): AnnotationImage =
    AnnotationImage(
      AnnotatorType.IMAGE,
      origin = image.origin,
      height = image.height,
      width = image.width,
      nChannels = image.nChannels,
      mode = image.mode,
      result = Array.emptyByteArray,
      Map.empty[String, String])

}
