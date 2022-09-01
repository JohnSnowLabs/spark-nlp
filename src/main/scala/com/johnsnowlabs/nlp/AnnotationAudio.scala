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

/** Represents [[AudioAssembler]]'s output parts and their details. */
case class AnnotationAudio(
    annotatorType: String,
    path: String,
    modificationTime: java.sql.Timestamp,
    length: Long,
    result: Array[Byte],
    metadata: Map[String, String]) {
  override def equals(obj: Any): Boolean = {
    obj match {
      case annotation: AnnotationAudio =>
        this.annotatorType == annotation.annotatorType &&
        this.path == annotation.path &&
        this.modificationTime == annotation.modificationTime &&
        this.length == annotation.length &&
        this.result.sameElements(annotation.result) &&
        this.metadata == annotation.metadata
      case _ => false
    }
  }
}

object AnnotationAudio {

  case class AnnotationContainer(__annotation: Array[AnnotationAudio])

  /** This is spark type of an annotation representing its metadata shape */
  val dataType = new StructType(
    Array(
      StructField("annotatorType", StringType, nullable = true),
      StructField("path", StringType, nullable = false),
      StructField("modificationTime", TimestampType, nullable = false),
      StructField("length", LongType, nullable = false),
      StructField("result", BinaryType, nullable = true),
      StructField("metadata", MapType(StringType, StringType), nullable = true)))

  val arrayType = new ArrayType(dataType, true)

  case class AudioFields(
      path: String,
      modificationTime: java.sql.Timestamp,
      length: Long,
      result: Array[Byte])

  def apply(row: Row): AnnotationAudio = {
    AnnotationAudio(
      row.getString(0),
      row.getAs[Row]("struct").getString(1),
      row.getAs[Row]("struct").getTimestamp(2),
      row.getAs[Row]("struct").getLong(3),
      row.getAs[Row]("struct").getSeq[Byte](4).toArray,
      row.getMap[String, String](5))
  }

  def apply(audio: AudioFields): AnnotationAudio =
    AnnotationAudio(
      AnnotatorType.AUDIO,
      path = audio.path,
      modificationTime = audio.modificationTime,
      length = audio.length,
      result = Array.emptyByteArray,
      Map.empty[String, String])

}
