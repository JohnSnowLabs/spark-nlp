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

import com.johnsnowlabs.nlp.util.{AnnotationRowUtils, SparkNlpConfig}
import com.johnsnowlabs.util.Version
import org.apache.spark.sql.api.java.UDF1
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{array, col, explode, udf}
import org.apache.spark.sql.types.{
  BooleanType,
  DataType,
  MetadataBuilder,
  StructField,
  StructType
}
import org.apache.spark.sql.{Column, DataFrame, Row}

import scala.reflect.runtime.universe.{typeOf, TypeTag}

object functions {

  private def isSpark4OrNewer(dataset: DataFrame): Boolean =
    Version.parse(dataset.sparkSession.version).toFloat >= 4.0f

  private def annotationsFromRows(rows: scala.collection.Seq[Row]): Seq[Annotation] = {
    Option(rows).map(_.map(Annotation(_)).toSeq).getOrElse(Seq.empty[Annotation])
  }

  private def annotationRowsFromFunction(
      function: Seq[Annotation] => Seq[Annotation],
      rows: scala.collection.Seq[Row]): scala.collection.Seq[Row] = {
    AnnotationRowUtils.annotationsToRows(function(annotationsFromRows(rows))).toVector
  }

  private def annotationUdf(function: Seq[Annotation] => Seq[Annotation]): UserDefinedFunction = {
    val func = new UDF1[scala.collection.Seq[Row], scala.collection.Seq[Row]] {
      override def call(rows: scala.collection.Seq[Row]): scala.collection.Seq[Row] = {
        annotationRowsFromFunction(function, rows)
      }
    }
    udf(func, Annotation.arrayType)
  }

  private def outputDataType[T: TypeTag]: DataType = {
    val tpe = typeOf[T]
    if (tpe <:< typeOf[Seq[Annotation]] || tpe <:< typeOf[Array[Annotation]]) {
      Annotation.arrayType
    } else {
      ScalaReflection.schemaFor[T].dataType
    }
  }

  private def normalizeOutput(value: Any, dataType: DataType): Any = {
    if (dataType == Annotation.arrayType) {
      value match {
        case annotations: scala.collection.Seq[_] =>
          AnnotationRowUtils
            .annotationsToRows(annotations.asInstanceOf[scala.collection.Seq[Annotation]])
            .toVector
        case annotations: Array[_] =>
          AnnotationRowUtils
            .annotationsToRows(annotations.toSeq.asInstanceOf[scala.collection.Seq[Annotation]])
            .toVector
        case null => scala.collection.Seq.empty[Row]
        case other => other
      }
    } else {
      value
    }
  }

  private def outputSchemaWithReplacement(
      inputSchema: StructType,
      outputCol: String,
      dataType: DataType,
      metadataBuilder: MetadataBuilder): StructType = {
    val outputField = StructField(outputCol, dataType, nullable = true, metadataBuilder.build())
    val outputIndex = inputSchema.fieldNames.indexOf(outputCol)

    if (outputIndex >= 0) StructType(inputSchema.fields.updated(outputIndex, outputField))
    else StructType(inputSchema.fields :+ outputField)
  }

  private def exactColumn(dataFrame: DataFrame, columnName: String): Column = {
    val escapedColumnName = columnName.replace("`", "``")
    dataFrame.col(s"`$escapedColumnName`")
  }

  private[nlp] def restoreSchemaMetadata(
      dataFrame: DataFrame,
      intendedSchema: StructType): DataFrame = {
    val columnsWithMetadata = intendedSchema.fields.map { field =>
      exactColumn(dataFrame, field.name).as(field.name, field.metadata)
    }
    dataFrame.select(columnsWithMetadata.toIndexedSeq: _*)
  }

  private def mapAnnotationsColWithRows[T: TypeTag](
      dataset: DataFrame,
      columns: Seq[String],
      outputCol: String,
      annotatorType: String,
      function: Seq[Annotation] => T): DataFrame = {
    val inputDataFrame = dataset.toDF()
    val metadataBuilder = new MetadataBuilder().putString("annotatorType", annotatorType)
    val dataType = outputDataType[T]
    val outputSchema =
      outputSchemaWithReplacement(inputDataFrame.schema, outputCol, dataType, metadataBuilder)
    val inputIndexes = columns.map(inputDataFrame.schema.fieldIndex)
    val outputIndex = inputDataFrame.schema.fieldNames.indexOf(outputCol)

    implicit val encoder: ExpressionEncoder[Row] =
      SparkNlpConfig.getEncoder(inputDataFrame, outputSchema)

    val mappedDataFrame = inputDataFrame
      .mapPartitions { rows =>
        rows.map { row =>
          val annotations = inputIndexes
            .flatMap(inputIndex => AnnotationRowUtils.extractAnnotationRows(row, inputIndex))
            .map(Annotation(_))
            .toSeq
          val outputValue = normalizeOutput(function(annotations), dataType)
          val outputValues =
            if (outputIndex >= 0) row.toSeq.updated(outputIndex, outputValue)
            else row.toSeq :+ outputValue
          Row.fromSeq(outputValues)
        }
      }
      .toDF()

    restoreSchemaMetadata(mappedDataFrame, outputSchema)
  }

  implicit class FilterAnnotations(dataset: DataFrame) {
    def filterByAnnotationsCol(
        column: String,
        function: Seq[Annotation] => Boolean): DataFrame = {
      if (isSpark4OrNewer(dataset)) {
        val inputDataFrame = dataset.toDF()
        val inputIndex = inputDataFrame.schema.fieldIndex(column)
        implicit val encoder: ExpressionEncoder[Row] =
          SparkNlpConfig.getEncoder(inputDataFrame, inputDataFrame.schema)

        val filteredDataFrame = inputDataFrame
          .mapPartitions { rows =>
            rows.filter { row =>
              val annotations =
                annotationsFromRows(AnnotationRowUtils.extractAnnotationRows(row, inputIndex))
              function(annotations)
            }
          }
          .toDF()

        restoreSchemaMetadata(filteredDataFrame, inputDataFrame.schema)
      } else {
        val meta = dataset.schema(column).metadata
        val func = udf { annotatorProperties: Seq[Row] =>
          function(annotationsFromRows(annotatorProperties))
        }
        dataset.filter(func(col(column)).as(column, meta))
      }
    }
  }

  def mapAnnotations(function: Seq[Annotation] => Seq[Annotation]): UserDefinedFunction =
    annotationUdf(function)

  def mapAnnotationsStrict(function: Seq[Annotation] => Seq[Annotation]): UserDefinedFunction =
    annotationUdf(function)

  implicit class MapAnnotations(dataset: DataFrame) {
    def mapAnnotationsCol[T: TypeTag](
        column: String,
        outputCol: String,
        annotatorType: String,
        function: Seq[Annotation] => T): DataFrame = {
      if (isSpark4OrNewer(dataset)) {
        mapAnnotationsColWithRows(dataset, Seq(column), outputCol, annotatorType, function)
      } else {
        val metadataBuilder: MetadataBuilder = new MetadataBuilder()
        val meta = metadataBuilder.putString("annotatorType", annotatorType).build()
        val func = udf { annotatorProperties: Seq[Row] =>
          function(annotationsFromRows(annotatorProperties))
        }
        dataset.withColumn(outputCol, func(col(column)).as(outputCol, meta))
      }
    }

    def mapAnnotationsCol[T: TypeTag](
        cols: Seq[String],
        outputCol: String,
        annotatorType: String,
        function: Seq[Annotation] => T): DataFrame = {
      if (isSpark4OrNewer(dataset)) {
        mapAnnotationsColWithRows(dataset, cols, outputCol, annotatorType, function)
      } else {
        val metadataBuilder: MetadataBuilder = new MetadataBuilder()
        val meta = metadataBuilder.putString("annotatorType", annotatorType).build()
        val func = udf { (cols: Seq[Seq[Row]]) =>
          function {
            cols.flatMap(aa => annotationsFromRows(aa))
          }
        }
        val inputCols = cols.map(col)
        dataset.withColumn(outputCol, func(array(inputCols: _*)).as(outputCol, meta))
      }
    }

  }

  implicit class EachAnnotations(dataset: DataFrame) {

    import dataset.sparkSession.implicits._

    def eachAnnotationsCol[T: TypeTag](
        column: String,
        function: Seq[Annotation] => Unit): Unit = {
      dataset.select(column).as[Array[Annotation]].foreach(function(_))
    }
  }

  implicit class ExplodeAnnotations(dataset: DataFrame) {
    def explodeAnnotationsCol[T: TypeTag](column: String, outputCol: String): DataFrame = {
      val meta = dataset.schema(column).metadata
      dataset
        .withColumn(outputCol, explode(col(column)))
        .withColumn(outputCol, array(col(outputCol)).as(outputCol, meta))
    }
  }
}
