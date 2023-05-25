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

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.{AnnotatorType, HasProtectedParams}
import org.apache.spark.ml.param.{IntParam, Params}
import org.apache.spark.sql.Column
import org.apache.spark.sql.types.MetadataBuilder

trait HasEmbeddingsProperties extends Params with HasProtectedParams {

  /** Number of embedding dimensions (Default depends on model)
    *
    * @group param
    */
  val dimension = new IntParam(this, "dimension", "Number of embedding dimensions").setProtected()

  /** @group setParam */
  def setDimension(value: Int): this.type = set(this.dimension, value)

  /** @group getParam */
  def getDimension: Int = $(dimension)

  protected def wrapEmbeddingsMetadata(
      col: Column,
      embeddingsDim: Int,
      embeddingsRef: Option[String] = None): Column = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", AnnotatorType.WORD_EMBEDDINGS)
    metadataBuilder.putLong("dimension", embeddingsDim.toLong)
    embeddingsRef.foreach(ref => metadataBuilder.putString("ref", ref))
    col.as(col.toString, metadataBuilder.build)
  }

  protected def wrapSentenceEmbeddingsMetadata(
      col: Column,
      embeddingsDim: Int,
      embeddingsRef: Option[String] = None): Column = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", AnnotatorType.SENTENCE_EMBEDDINGS)
    metadataBuilder.putLong("dimension", embeddingsDim.toLong)
    embeddingsRef.foreach(ref => metadataBuilder.putString("ref", ref))
    col.as(col.toString, metadataBuilder.build)
  }

}
