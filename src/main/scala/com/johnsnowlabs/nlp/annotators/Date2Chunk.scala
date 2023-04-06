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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/** Converts `DATE` type Annotations to `CHUNK` type.
  *
  * This can be useful if the following annotators after DateMatcher and MultiDateMatcher require
  * `CHUNK` types. The entity name in the metadata can be changed with `setEntityName`.
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.annotator._
  *
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val inputFormats = Array("yyyy", "yyyy/dd/MM", "MM/yyyy", "yyyy")
  * val outputFormat = "yyyy/MM/dd"
  *
  * val date = new DateMatcher()
  *   .setInputCols("document")
  *   .setOutputCol("date")
  *
  *
  * val date2Chunk = new Date2Chunk()
  *   .setInputCols("date")
  *   .setOutputCol("date_chunk")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   date,
  *   date2Chunk
  * ))
  *
  * val data = Seq(
  * """Omicron is a new variant of COVID-19, which the World Health Organization designated a variant of concern on Nov. 26, 2021/26/11.""",
  * """Neighbouring Austria has already locked down its population this week for at until 2021/10/12, becoming the first to reimpose such restrictions."""
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.transform(data).select("date_chunk").show(false)
  * ----------------------------------------------------+
  * date_chunk                                          |
  * ----------------------------------------------------+
  * [{chunk, 118, 121, 2021/01/01, {sentence -> 0}, []}]|
  * [{chunk, 83, 86, 2021/01/01, {sentence -> 0}, []}]  |
  * ----------------------------------------------------+
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
class Date2Chunk(override val uid: String)
    extends AnnotatorModel[Date2Chunk]
    with HasSimpleAnnotate[Date2Chunk] {

  /** Output Annotator Type : CHUNK
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = CHUNK

  /** Input Annotator Type : DATE
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DATE)

  def this() = this(Identifiable.randomUID("Date2Chunk"))

  /** Entity name for metadata (Default: `DATE`)
    *
    * @group param
    */
  val entityName =
    new Param[String](this, "entityName", "Entity name for the metadata")

  /** Sets entity name for metadata (Default: `DATE`)
    *
    * @group setParam
    */
  def setEntityName(name: String): this.type = set(this.entityName, name)

  setDefault(entityName -> "DATE")
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map { date =>
      Annotation(
        CHUNK,
        date.begin,
        date.end,
        date.result,
        date.metadata ++ Map("entity" -> $(entityName), "chunk" -> "0"))
    }
  }

}

/** This is the companion object of [[Date2Chunk]]. Please refer to that class for the
  * documentation.
  */
object Date2Chunk extends DefaultParamsReadable[Date2Chunk]
