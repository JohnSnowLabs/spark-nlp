/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.ner

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, NAMED_ENTITY, TOKEN}
import com.johnsnowlabs.nlp.annotators.common.NerTagged
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, HasSimpleAnnotate, ParamsAndFeaturesReadable}

import org.apache.spark.ml.param.{BooleanParam, StringArrayParam}
import org.apache.spark.ml.util.Identifiable

import scala.collection.immutable.Map

/**
 * Converts IOB or IOB2 representation of NER to user-friendly.
 * See https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
 *
 * @groupname anno Annotator types
 * @groupdesc anno Required input and expected output annotator types
 * @groupname Ungrouped Members
 * @groupname param Parameters
 * @groupname setParam Parameter setters
 * @groupname getParam Parameter getters
 * @groupname Ungrouped Members
 * @groupprio param  1
 * @groupprio anno  2
 * @groupprio Ungrouped 3
 * @groupprio setParam  4
 * @groupprio getParam  5
 * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
 */
class NerConverter(override val uid: String) extends AnnotatorModel[NerConverter] with HasSimpleAnnotate[NerConverter] {

  def this() = this(Identifiable.randomUID("NER_CONVERTER"))

  /** Input Annotator Type : DOCUMENT, TOKEN, NAMED_ENTITY
   *
   * @group anno
   **/
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN, NAMED_ENTITY)
  /** Output Annotator Type : CHUNK
   *
   * @group anno
   **/
  override val outputAnnotatorType: AnnotatorType = CHUNK

  /** If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix on labels
   *
   * @group param
   **/
  val whiteList: StringArrayParam = new StringArrayParam(this, "whiteList", "If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix on labels")

  /** If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix on labels
   *
   * @group setParam
   **/
  def setWhiteList(list: String*): NerConverter.this.type = set(whiteList, list.toArray)

  /** Whether to preserve the original position of the tokens in the original document or use the modified tokens
   *
   * @group param
   **/
  val preservePosition: BooleanParam = new BooleanParam(this, "preservePosition", "Whether to preserve the original position of the tokens in the original document or use the modified tokens")

  /** Whether to preserve the original position of the tokens in the original document or use the modified tokens
   *
   * @group setParam
   **/
  def setPreservePosition(value: Boolean): this.type = set(preservePosition, value)

  setDefault(
    preservePosition -> true
  )

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = NerTagged.unpack(annotations)
    val docs = annotations.filter(a => a.annotatorType == AnnotatorType.DOCUMENT && sentences.exists(
      b => b.indexedTaggedWords.exists(c => c.begin >= a.begin && c.end <= a.end)
    ))

    val entities = sentences.zip(docs.zipWithIndex).flatMap { case (sentence, doc) =>
      NerTagsEncoding.fromIOB(sentence, doc._1, sentenceIndex = doc._2, $(preservePosition))
    }

    entities.filter(entity => get(whiteList).forall(validEntity => validEntity.contains(entity.entity))).
      zipWithIndex.map{case (entity, idx) =>
      val baseMetadata = Map("entity" -> entity.entity, "sentence" -> entity.sentenceId, "chunk" -> idx.toString)
      val metadata = if(entity.confidence.isEmpty) baseMetadata else baseMetadata + ("confidence" -> entity.confidence.get.toString)
      Annotation(
        outputAnnotatorType,
        entity.start,
        entity.end,
        entity.text,
        metadata
      )

    }
  }

}

object NerConverter extends ParamsAndFeaturesReadable[NerConverter]
