/*
 * Copyright 2017-2026 John Snow Labs
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

import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.StructType

/** Merges multiple annotation columns into a single annotation column.
  *
  * This is useful when multiple annotators produce separate annotation columns (e.g.,
  * `document_text`, `document_table` from `ReaderAssembler`) and a downstream annotator (e.g.,
  * `AutoGGUFVisionModel`) expects a single input column containing all annotations.
  *
  * Annotations from all input columns are collected and concatenated into the output column. The
  * output annotator type defaults to `DOCUMENT` but can be configured. Each annotation's metadata
  * is preserved, and a `source_column` key is added to track the original column name.
  *
  * '''Note:''' All input columns must use the `Annotation` schema (i.e., `Annotation.dataType`).
  * Columns using `AnnotationImage.dataType` (e.g., IMAGE-typed columns from `ReaderAssembler`)
  * are not supported and will cause a validation error.
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.{MultiColumnAssembler, DocumentAssembler}
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler1 = new DocumentAssembler()
  *   .setInputCol("text1")
  *   .setOutputCol("document_text")
  *
  * val documentAssembler2 = new DocumentAssembler()
  *   .setInputCol("text2")
  *   .setOutputCol("document_table")
  *
  * val multiColumnAssembler = new MultiColumnAssembler()
  *   .setInputCols("document_text", "document_table")
  *   .setOutputCol("merged_document")
  *
  * val data = Seq(("Hello world", "Name | Age\nJohn | 30"))
  *   .toDF("text1", "text2")
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(documentAssembler1, documentAssembler2, multiColumnAssembler))
  *   .fit(data)
  *
  * val result = pipeline.transform(data)
  * result.selectExpr("merged_document.result").show(false)
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class MultiColumnAssembler(override val uid: String)
    extends AnnotatorModel[MultiColumnAssembler]
    with HasSimpleAnnotate[MultiColumnAssembler]
    with HasMultipleInputAnnotationCols {

  import com.johnsnowlabs.nlp.AnnotatorType._

  def this() = this(Identifiable.randomUID("ANNOTATION_MERGER"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    *
    * @group anno
    */
  override val inputAnnotatorType: String = DOCUMENT

  /** Output annotator types: DOCUMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  /** The annotator type to use for the output column (Default: `document`).
    *
    * This can be changed to match the expected input type of downstream annotators.
    *
    * @group param
    */
  val outputAsAnnotatorType: Param[String] = new Param[String](
    this,
    "outputAsAnnotatorType",
    "The annotator type to use for the output annotations (Default: `document`)")

  /** Whether to sort merged annotations by their begin position (Default: `false`).
    *
    * When `false`, annotations are ordered by input column order (all from first column, then
    * second, etc.). When `true`, annotations from all columns are interleaved by their `begin`
    * position.
    *
    * @group param
    */
  val sortByBegin: BooleanParam = new BooleanParam(
    this,
    "sortByBegin",
    "Whether to sort merged annotations by their begin position (Default: `false`)")

  setDefault(outputAsAnnotatorType -> DOCUMENT, sortByBegin -> false)

  /** @group setParam */
  def setOutputAsAnnotatorType(value: String): this.type = set(outputAsAnnotatorType, value)

  /** @group getParam */
  def getOutputAsAnnotatorType: String = $(outputAsAnnotatorType)

  /** @group setParam */
  def setSortByBegin(value: Boolean): this.type = set(sortByBegin, value)

  /** @group getParam */
  def getSortByBegin: Boolean = $(sortByBegin)

  /** Override validation to accept any annotation type on input columns, not just DOCUMENT. This
    * allows merging columns with different annotator types (e.g., DOCUMENT + TABLE). Columns must
    * still have the `annotatorType` metadata key. IMAGE columns are rejected because they use
    * `AnnotationImage.dataType` which is incompatible with the standard `Annotation` schema.
    */
  override protected def validate(schema: StructType): Boolean = {
    getInputCols.forall { colName =>
      schema.exists { field =>
        field.name == colName &&
        field.metadata.contains("annotatorType") &&
        field.metadata.getString("annotatorType") != AnnotatorType.IMAGE
      }
    }
  }

  override protected def extraValidateMsg: String =
    s"MultiColumnAssembler input columns must have annotation metadata and cannot be IMAGE type. " +
      s"Current inputCols: ${getInputCols.mkString(", ")}"

  override protected def extraValidate(structType: StructType): Boolean = {
    getInputCols.forall { colName =>
      structType.exists { field =>
        field.name == colName &&
        field.metadata.contains("annotatorType") &&
        field.metadata.getString("annotatorType") != AnnotatorType.IMAGE
      }
    }
  }

  /** Takes annotations from all input columns and merges them into a single sequence.
    *
    * This is a fallback used when dfAnnotate is not invoked directly. In normal pipeline
    * execution, dfAnnotate is used instead so that source column names can be tracked.
    *
    * @param annotations
    *   Annotations from all input columns, flattened
    * @return
    *   Merged annotations with output annotator type applied
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val targetType = $(outputAsAnnotatorType)
    val merged = annotations.map { annotation =>
      annotation.copy(
        annotatorType = targetType,
        metadata = annotation.metadata + ("source_column" -> "unknown"))
    }
    if ($(sortByBegin)) merged.sortBy(_.begin)
    else merged
  }

  /** Overrides the default dfAnnotate to preserve per-column source information.
    *
    * The default `HasSimpleAnnotate.dfAnnotate` flattens all input columns before calling
    * `annotate`, losing track of which annotation came from which column. This override zips the
    * per-column annotation sequences with `getInputCols` to tag each annotation's metadata with
    * its true `source_column` name.
    */
  override def dfAnnotate: UserDefinedFunction = {
    val inputCols = getInputCols
    val targetType = $(outputAsAnnotatorType)
    val doSort = $(sortByBegin)

    udf { annotationProperties: Seq[AnnotationContent] =>
      val merged = annotationProperties.zip(inputCols).flatMap { case (colAnnotations, colName) =>
        colAnnotations.map { row =>
          val annotation = Annotation(row)
          annotation.copy(
            annotatorType = targetType,
            metadata = annotation.metadata + ("source_column" -> colName))
        }
      }
      if (doSort) merged.sortBy(_.begin)
      else merged
    }
  }

}

/** This is the companion object of [[MultiColumnAssembler]]. Please refer to that class for the
  * documentation.
  */
object MultiColumnAssembler extends DefaultParamsReadable[MultiColumnAssembler]
