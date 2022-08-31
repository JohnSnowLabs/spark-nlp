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

package com.johnsnowlabs.nlp.audio

import org.apache.spark.sql.types._

import scala.collection.Map

/** Represents [[AudioAssembler]]'s output parts and their details. */
case class AnnotationAudio(
    annotatorType: String,
    result: Array[Float],
    metadata: Map[String, String]) {
  override def equals(obj: Any): Boolean = {
    obj match {
      case annotation: AnnotationAudio =>
        this.annotatorType == annotation.annotatorType &&
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
      StructField("origin", StringType, nullable = false),
      StructField("length", IntegerType, nullable = false),
      StructField("samplingRate", IntegerType, nullable = false),
      StructField("result", BinaryType, nullable = false),
      StructField("metadata", MapType(StringType, StringType), nullable = true)))

//  val arrayType = new ArrayType(dataType, true)

//  case class AudioFields(
//                          origin: String,
//                          length: Int,
//                          sampleRate: Int,
//                          data: Array[Float],
//                          text: Option[String])

//  /** TODO: needed?
//    *
//    * @param row
//    *   spark row to be converted
//    * @return
//    *   AnnotationAudio
//    */
//  def apply(row: Row): AnnotationAudio = {
//    AnnotationAudio(
//      row.getString(0),
//      row.getString(1),
//      row.getInt(2),
//      row.getInt(3),
//      row.getInt(4),
//      row.getInt(5),
//      row.getSeq[Byte](6).toArray,
//      row.getMap[String, String](7))
//  }

  def apply(rawAudio: Array[Byte]): AnnotationAudio = {
    AudioProcessors.createAnnotationAudio(rawAudio)
  }

}
