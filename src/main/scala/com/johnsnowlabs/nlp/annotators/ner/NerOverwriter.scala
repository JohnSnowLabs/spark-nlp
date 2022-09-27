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

import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import org.apache.spark.ml.param.{Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import java.util
import scala.collection.JavaConverters._

/** Overwrites entities of specified strings.
  *
  * The input for this Annotator have to be entities that are already extracted, Annotator type
  * `NAMED_ENTITY`. The strings specified with `setStopWords` will have new entities assigned to,
  * specified with `setNewResult`.
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
  * import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
  * import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
  * import com.johnsnowlabs.nlp.annotators.ner.NerOverwriter
  * import org.apache.spark.ml.Pipeline
  *
  * // First extract the prerequisite Entities
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentence = new SentenceDetector()
  *   .setInputCols("document")
  *   .setOutputCol("sentence")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("sentence")
  *   .setOutputCol("token")
  *
  * val embeddings = WordEmbeddingsModel.pretrained()
  *   .setInputCols("sentence", "token")
  *   .setOutputCol("embeddings")
  *
  * val nerTagger = NerDLModel.pretrained()
  *   .setInputCols("sentence", "token", "embeddings")
  *   .setOutputCol("ner")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   sentence,
  *   tokenizer,
  *   embeddings,
  *   nerTagger
  * ))
  *
  * val data = Seq("Spark NLP Crosses Five Million Downloads, John Snow Labs Announces.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(ner)").show(false)
  * /*
  * +------------------------------------------------------+
  * |col                                                   |
  * +------------------------------------------------------+
  * |[named_entity, 0, 4, B-ORG, [word -> Spark], []]      |
  * |[named_entity, 6, 8, I-ORG, [word -> NLP], []]        |
  * |[named_entity, 10, 16, O, [word -> Crosses], []]      |
  * |[named_entity, 18, 21, O, [word -> Five], []]         |
  * |[named_entity, 23, 29, O, [word -> Million], []]      |
  * |[named_entity, 31, 39, O, [word -> Downloads], []]    |
  * |[named_entity, 40, 40, O, [word -> ,], []]            |
  * |[named_entity, 42, 45, B-ORG, [word -> John], []]     |
  * |[named_entity, 47, 50, I-ORG, [word -> Snow], []]     |
  * |[named_entity, 52, 55, I-ORG, [word -> Labs], []]     |
  * |[named_entity, 57, 65, I-ORG, [word -> Announces], []]|
  * |[named_entity, 66, 66, O, [word -> .], []]            |
  * +------------------------------------------------------+
  * */
  * // The recognized entities can then be overwritten
  * val nerOverwriter = new NerOverwriter()
  *   .setInputCols("ner")
  *   .setOutputCol("ner_overwritten")
  *   .setNerWords(Array("Million"))
  *   .setNerNewEntity,("B-CARDINAL")
  *
  * nerOverwriter.transform(result).selectExpr("explode(ner_overwritten)").show(false)
  * +---------------------------------------------------------+
  * |col                                                      |
  * +---------------------------------------------------------+
  * |[named_entity, 0, 4, B-ORG, [word -> Spark], []]         |
  * |[named_entity, 6, 8, I-ORG, [word -> NLP], []]           |
  * |[named_entity, 10, 16, O, [word -> Crosses], []]         |
  * |[named_entity, 18, 21, O, [word -> Five], []]            |
  * |[named_entity, 23, 29, B-CARDINAL, [word -> Million], []]|
  * |[named_entity, 31, 39, O, [word -> Downloads], []]       |
  * |[named_entity, 40, 40, O, [word -> ,], []]               |
  * |[named_entity, 42, 45, B-ORG, [word -> John], []]        |
  * |[named_entity, 47, 50, I-ORG, [word -> Snow], []]        |
  * |[named_entity, 52, 55, I-ORG, [word -> Labs], []]        |
  * |[named_entity, 57, 65, I-ORG, [word -> Announces], []]   |
  * |[named_entity, 66, 66, O, [word -> .], []]               |
  * +---------------------------------------------------------+
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
class NerOverwriter(override val uid: String)
    extends AnnotatorModel[NerOverwriter]
    with HasSimpleAnnotate[NerOverwriter] {

  import com.johnsnowlabs.nlp.AnnotatorType.NAMED_ENTITY

  /** Output Annotator Type : NAMED_ENTITY
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = NAMED_ENTITY

  /** Input Annotator Type : NAMED_ENTITY
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(NAMED_ENTITY)

  def this() = this(Identifiable.randomUID("NER_OVERWRITER"))

  /** The words to be filtered out.
    *
    * @group param
    */
  val nerWords: StringArrayParam =
    new StringArrayParam(
      this,
      "nerWords",
      "The words to be filtered out to replace the current NER entity with a new NER Entity")

  /** The words to be filtered out.
    *
    * @group setParam
    */
  def setNerWords(value: Array[String]): this.type = set(nerWords, value)

  /** The words to be filtered out.
    *
    * @group getParam
    */
  def getNerWords: Array[String] = $(nerWords)

  /** New NER class to overwrite
    *
    * @group param
    */
  val newNerEntity: Param[String] = new Param(this, "newNerEntity", "New NER label to overwrite")

  /** New NER class to overwrite
    *
    * @group setParam
    */
  def setNewNerEntity(r: String): this.type = {
    set(newNerEntity, r)
  }

  /** New NER class to overwrite
    *
    * @group getParam
    */
  def getNewNerEntity: String = $(newNerEntity)

  val replaceEntities: MapFeature[String, String] =
    new MapFeature[String, String](this, "replaceEntities")

  def setReplaceEntities(w: Map[String, String]): this.type = set(replaceEntities, w)

  // for Python access
  /** @group setParam */
  def setReplaceEntities(w: util.HashMap[String, String]): this.type = {

    val ws = w.asScala.toMap
    set(replaceEntities, ws)
  }

  def getReplaceEntities: Map[String, String] = {
    if (!replaceEntities.isSet) {
      Map.empty[String, String]
    } else {
      $$(replaceEntities)
    }
  }

  setDefault(newNerEntity -> "I-OVERWRITE", nerWords -> Array())

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val annotationsOverwritten = annotations
    val replace = getReplaceEntities
    annotationsOverwritten
      .map { tokenAnnotation =>
        val nerWordsSet = $(nerWords).toSet
        if (nerWordsSet.contains(tokenAnnotation.metadata("word"))) {
          Annotation(
            outputAnnotatorType,
            tokenAnnotation.begin,
            tokenAnnotation.end,
            $(newNerEntity),
            tokenAnnotation.metadata)
        } else {
          Annotation(
            outputAnnotatorType,
            tokenAnnotation.begin,
            tokenAnnotation.end,
            tokenAnnotation.result,
            tokenAnnotation.metadata)
        }
      }
      .map { ann =>
        ann.copy(result = replace.getOrElse(ann.result, ann.result))
      }

  }

}

/** This is the companion object of [[NerOverwriter]]. Please refer to that class for the
  * documentation.
  */
object NerOverwriter extends DefaultParamsReadable[NerOverwriter]
