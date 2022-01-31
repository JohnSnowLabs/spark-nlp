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

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.{array, col, explode, udf}
import org.apache.spark.sql.types.MetadataBuilder
import scala.reflect.runtime.universe._


object functions {

  type JSLAnnotation = com.johnsnowlabs.nlp.Annotation

  implicit class FilterAnnotations(dataset: DataFrame) {
    def filterByAnnotationsCol(column: String, function: collection.Seq[JSLAnnotation] => Boolean): DataFrame = {
      val meta = dataset.schema(column).metadata
      val func = udf {
        annotatorProperties: collection.Seq[Row] => function(annotatorProperties.map(com.johnsnowlabs.nlp.Annotation(_)))
      }
      dataset.filter(func(col(column)).as(column, meta))
    }
  }

  def mapAnnotations(function: Seq[JSLAnnotation] => Seq[JSLAnnotation]): UserDefinedFunction =
    udf { annotatorProperties: Seq[Row] => function(annotatorProperties.map(com.johnsnowlabs.nlp.Annotation(_))) }

  // FIXME eliminate or create facade or declare one as deprecated?
  def mapAnnotationsStrict(function: Seq[JSLAnnotation] => Seq[JSLAnnotation]): UserDefinedFunction =
    udf { annotatorProperties: Seq[Row] => function(annotatorProperties.map(com.johnsnowlabs.nlp.Annotation(_))) }

  implicit class MapAnnotations(dataset: DataFrame) {
    def mapAnnotationsCol[T: TypeTag](column: String, outputCol: String, annotatorType: String, function: Seq[JSLAnnotation] => T): DataFrame = {
      val metadataBuilder: MetadataBuilder = new MetadataBuilder()
      val meta = metadataBuilder.putString("annotatorType", annotatorType).build()
      val func = udf {
        annotatorProperties: Seq[Row] => function(annotatorProperties.map(com.johnsnowlabs.nlp.Annotation(_)))
      }
      dataset.withColumn(outputCol, func(col(column)).as(outputCol, meta))
    }

    def mapAnnotationsCol[T: TypeTag](cols: Seq[String], outputCol: String, annotatorType: String, function: Seq[JSLAnnotation] => T): DataFrame = {
      val metadataBuilder: MetadataBuilder = new MetadataBuilder()
      val meta = metadataBuilder.putString("annotatorType", annotatorType).build()
      val func = udf {
        (cols: Seq[Seq[Row]]) => function { cols.flatMap(aa => aa.map(com.johnsnowlabs.nlp.Annotation(_))) }
      }
      val inputCols = cols.map(col)
      dataset.withColumn(outputCol, func(array(inputCols: _*)).as(outputCol, meta))
    }
  }


  implicit class EachAnnotations(dataset: DataFrame) {

    import dataset.sparkSession.implicits._

    def eachAnnotationsCol[T: TypeTag](column: String, function: Seq[JSLAnnotation] => Unit): Unit = {
      dataset.select(column).as[Array[JSLAnnotation]].foreach(function(_))
    }
  }

  implicit class ExplodeAnnotations(dataset: DataFrame) {
    def explodeAnnotationsCol[T: TypeTag](column: String, outputCol: String): DataFrame = {
      val meta = dataset.schema(column).metadata
      dataset.
        withColumn(outputCol, explode(col(column))).
        withColumn(outputCol, array(col(outputCol)).as(outputCol, meta))
    }
  }
}
