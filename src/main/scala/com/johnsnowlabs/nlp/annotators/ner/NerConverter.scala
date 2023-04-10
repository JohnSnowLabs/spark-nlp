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

package com.johnsnowlabs.nlp.annotators.ner

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, NAMED_ENTITY, TOKEN}
import com.johnsnowlabs.nlp.annotators.common.NerTagged
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.param.{BooleanParam, StringArrayParam}
import org.apache.spark.ml.util.Identifiable

import scala.collection.immutable.Map

/** Converts a IOB or IOB2 representation of NER to a user-friendly one, by associating the tokens
  * of recognized entities and their label. Results in `CHUNK` Annotation type.
  *
  * NER chunks can then be filtered by setting a whitelist with `setWhiteList`. Chunks with no
  * associated entity (tagged "O") are filtered.
  *
  * See also
  * [[https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging) Inside–outside–beginning (tagging)]]
  * for more information.
  *
  * ==Example==
  * This is a continuation of the example of the
  * [[com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel NerDLModel]]. See that class on how to
  * extract the entities.
  *
  * The output of the NerDLModel follows the Annotator schema and can be converted like so:
  * {{{
  * result.selectExpr("explode(ner)").show(false)
  * +----------------------------------------------------+
  * |col                                                 |
  * +----------------------------------------------------+
  * |[named_entity, 0, 2, B-ORG, [word -> U.N], []]      |
  * |[named_entity, 3, 3, O, [word -> .], []]            |
  * |[named_entity, 5, 12, O, [word -> official], []]    |
  * |[named_entity, 14, 18, B-PER, [word -> Ekeus], []]  |
  * |[named_entity, 20, 24, O, [word -> heads], []]      |
  * |[named_entity, 26, 28, O, [word -> for], []]        |
  * |[named_entity, 30, 36, B-LOC, [word -> Baghdad], []]|
  * |[named_entity, 37, 37, O, [word -> .], []]          |
  * +----------------------------------------------------+
  * }}}
  * After the converter is used:
  * {{{
  * val converter = new NerConverter()
  *   .setInputCols("sentence", "token", "ner")
  *   .setOutputCol("entities")
  *   .setPreservePosition(false)
  *
  * converter.transform(result).selectExpr("explode(entities)").show(false)
  * +------------------------------------------------------------------------+
  * |col                                                                     |
  * +------------------------------------------------------------------------+
  * |[chunk, 0, 2, U.N, [entity -> ORG, sentence -> 0, chunk -> 0], []]      |
  * |[chunk, 14, 18, Ekeus, [entity -> PER, sentence -> 0, chunk -> 1], []]  |
  * |[chunk, 30, 36, Baghdad, [entity -> LOC, sentence -> 0, chunk -> 2], []]|
  * +------------------------------------------------------------------------+
  * }}}
  *
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
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
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class NerConverter(override val uid: String)
    extends AnnotatorModel[NerConverter]
    with HasSimpleAnnotate[NerConverter] {

  def this() = this(Identifiable.randomUID("NER_CONVERTER"))

  /** Input Annotator Type : DOCUMENT, TOKEN, NAMED_ENTITY
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN, NAMED_ENTITY)

  /** Output Annotator Type : CHUNK
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = CHUNK

  /** If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix
    * on labels
    *
    * @group param
    */
  val whiteList: StringArrayParam = new StringArrayParam(
    this,
    "whiteList",
    "If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix on labels")

  /** If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix
    * on labels
    *
    * @group setParam
    */
  def setWhiteList(list: String*): NerConverter.this.type = set(whiteList, list.toArray)

  /** Whether to preserve the original position of the tokens in the original document or use the
    * modified tokens (Default: `true`)
    *
    * @group param
    */
  val preservePosition: BooleanParam = new BooleanParam(
    this,
    "preservePosition",
    "Whether to preserve the original position of the tokens in the original document or use the modified tokens")

  /** Whether to preserve the original position of the tokens in the original document or use the
    * modified tokens (Default: `true`)
    *
    * @group setParam
    */
  def setPreservePosition(value: Boolean): this.type = set(preservePosition, value)

  /** set this to true if your NER tags coming from a model that does not have a IOB/IOB2 schema
    *
    * @group param
    */
  val nerHasNoSchema: BooleanParam = new BooleanParam(
    this,
    "nerHasNoSchema",
    "set this to true if your NER tags coming from a model that does not have a IOB/IOB2 schema")

  /** @group setParam */
  def setNerHasNoSchema(value: Boolean): this.type = set(nerHasNoSchema, value)

  setDefault(preservePosition -> true, nerHasNoSchema -> false)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = NerTagged.unpack(annotations)
    val docs = annotations.filter(a =>
      a.annotatorType == AnnotatorType.DOCUMENT && sentences.exists(b =>
        b.indexedTaggedWords.exists(c => c.begin >= a.begin && c.end <= a.end)))

    val entities = sentences.zip(docs.zipWithIndex).flatMap { case (sentence, doc) =>
      NerTagsEncoding.fromIOB(
        sentence,
        doc._1,
        sentenceIndex = doc._2,
        originalOffset = $(preservePosition),
        nerHasNoSchema = $(nerHasNoSchema))
    }

    entities
      .filter(entity => get(whiteList).forall(validEntity => validEntity.contains(entity.entity)))
      .zipWithIndex
      .map { case (entity, idx) =>
        val baseMetadata =
          Map("entity" -> entity.entity, "sentence" -> entity.sentenceId, "chunk" -> idx.toString)
        val metadata =
          if (entity.confidence.isEmpty) baseMetadata
          else baseMetadata + ("confidence" -> entity.confidence.get.toString)
        Annotation(outputAnnotatorType, entity.start, entity.end, entity.text, metadata)

      }
  }

}

object NerConverter extends ParamsAndFeaturesReadable[NerConverter]
