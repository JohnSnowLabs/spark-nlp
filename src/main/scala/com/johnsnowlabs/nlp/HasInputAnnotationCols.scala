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

import org.apache.spark.ml.param.{Params, StringArrayParam}
import org.apache.spark.sql.types.StructType

trait HasInputAnnotationCols extends Params {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  val inputAnnotatorTypes: Array[String]

  val optionalInputAnnotatorTypes: Array[String] = Array()

  /** columns that contain annotations necessary to run this annotator AnnotatorType is used both
    * as input and output columns if not specified
    */
  protected final val inputCols: StringArrayParam =
    new StringArrayParam(this, "inputCols", "the input annotation columns")

  /** Overrides required annotators column if different than default */
  def setInputCols(value: Array[String]): this.type = {
    if (optionalInputAnnotatorTypes.isEmpty) {
      require(
        value.length == inputAnnotatorTypes.length,
        s"setInputCols in ${this.uid} expecting ${inputAnnotatorTypes.length} columns. " +
          s"Provided column amount: ${value.length}. " +
          s"Which should be columns from the following annotators: ${inputAnnotatorTypes.mkString(", ")} ")
    } else {
      val expectedColumns = inputAnnotatorTypes.length + optionalInputAnnotatorTypes.length
      require(
        value.length == inputAnnotatorTypes.length || value.length == expectedColumns,
        s"setInputCols in ${this.uid} expecting at least ${inputAnnotatorTypes.length} columns. " +
          s"Provided column amount: ${value.length}. " +
          s"Which should be columns from at least the following annotators: ${inputAnnotatorTypes
              .mkString(", ")} ")
    }
    set(inputCols, value)
  }

  protected def msgHelper(schema: StructType): String = {
    val schemaInfo = schema.map(sc =>
      (
        "column_name=" + sc.name,
        "is_nlp_annotator=" + sc.metadata.contains("annotatorType") + {
          if (sc.metadata.contains("annotatorType"))
            ",type=" + sc.metadata.getString("annotatorType")
          else ""
        }))
    s"\nCurrent inputCols: ${getInputCols.mkString(",")}. Dataset's columns:\n${schemaInfo.mkString("\n")}."
  }

  final protected def checkSchema(schema: StructType, inputAnnotatorType: String): Boolean = {
    schema.exists { field =>
      {
        field.metadata.contains("annotatorType") &&
        field.metadata.getString("annotatorType") == inputAnnotatorType &&
        getInputCols.contains(field.name)
      }
    }
  }

  final def setInputCols(value: String*): this.type = setInputCols(value.toArray)

  /** @return input annotations columns currently used */
  def getInputCols: Array[String] =
    get(inputCols)
      .orElse(getDefault(inputCols))
      .getOrElse(throw new Exception(s"inputCols not provided." +
        s" Requires columns for ${inputAnnotatorTypes.mkString(", ")} annotators"))
}
