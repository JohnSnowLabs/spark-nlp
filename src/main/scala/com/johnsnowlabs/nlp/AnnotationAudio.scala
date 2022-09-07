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
    result: Array[Byte],
    metadata: Map[String, String])
    extends IAnnotation {
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
      StructField("result", ArrayType(ByteType, containsNull = false), nullable = true),
      StructField("metadata", MapType(StringType, StringType), nullable = true)))

  val arrayType = new ArrayType(dataType, true)

  case class AudioFields(result: Array[Byte])

  def apply(row: Row): AnnotationAudio = {
    AnnotationAudio(row.getString(0), row.getSeq[Byte](1).toArray, row.getMap[String, String](2))
  }

  def apply(audio: AudioFields): AnnotationAudio =
    AnnotationAudio(AnnotatorType.AUDIO, result = Array.emptyByteArray, Map.empty[String, String])

}
