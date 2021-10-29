/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.nlp.Annotation
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{ArrayType, MapType, StringType, StructField, StructType}

object FinisherUtil {

  def checkIfInputColsExist(inputCols: Array[String], schema: StructType): Unit = {
    require(inputCols.forall(schema.fieldNames.contains),
      s"pipeline annotator stages incomplete. " +
        s"expected: ${inputCols.mkString(", ")}, " +
        s"found: ${schema.fields.filter(_.dataType == ArrayType(Annotation.dataType)).map(_.name).mkString(", ")}, " +
        s"among available: ${schema.fieldNames.mkString(", ")}")
  }

  def checkIfAnnotationColumnIsSparkNLPAnnotation(schema: StructType, annotationColumn: String): Unit = {
    require(schema(annotationColumn).dataType == ArrayType(Annotation.dataType),
      s"column [$annotationColumn] must be an NLP Annotation column")
  }

  def getMetadataFields(outputCols: Array[String], outputAsArray: Boolean): Array[StructField] = {
    outputCols.flatMap(outputCol => {
      if (outputAsArray)
        Some(StructField(outputCol + "_metadata", MapType(StringType, StringType), nullable = false))
      else
        None
    })
  }

  def getOutputFields(outputCols: Array[String], outputAsArray: Boolean): Array[StructField] = {
    outputCols.map(outputCol => {
      if (outputAsArray)
        StructField(outputCol, ArrayType(StringType), nullable = false)
      else
        StructField(outputCol, StringType, nullable = true)
    })
  }

  def getCleanFields(cleanAnnotations: Boolean, outputFields: Array[StructField]): Array[StructField] = {
    if (cleanAnnotations) outputFields.filterNot(f =>
      f.dataType == ArrayType(Annotation.dataType)
    ) else outputFields
  }

  def cleaningAnnotations(cleanAnnotations: Boolean, dataSet: DataFrame): DataFrame = {
    if (cleanAnnotations) {
      val columnsToDrop = dataSet.schema.fields
        .filter(_.dataType == ArrayType(Annotation.dataType))
        .map(_.name)
      dataSet.drop(columnsToDrop:_*)
    } else dataSet
  }

}
