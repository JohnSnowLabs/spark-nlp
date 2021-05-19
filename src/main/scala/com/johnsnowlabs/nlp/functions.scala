package com.johnsnowlabs.nlp

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.{array, col, explode, udf}
import org.apache.spark.sql.types.DataType

import scala.reflect.runtime.universe._

object functions {

  implicit class FilterAnnotations(dataset: DataFrame) {
    def filterByAnnotationsCol(column: String, function: Seq[Annotation] => Boolean): DataFrame = {
      val meta = dataset.schema(column).metadata
      val func = udf {
        annotatorProperties: Seq[Row] =>
          function(annotatorProperties.map(Annotation(_)))
      }
      dataset.filter(func(col(column)).as(column, meta))
    }
  }

  def mapAnnotations(function: Seq[Annotation] => Seq[Annotation]): UserDefinedFunction =
    udf { annotatorProperties: Seq[Row] => function(annotatorProperties.map(Annotation(_))) }

  def mapAnnotationsStrict(function: Seq[Annotation] => Seq[Annotation]): UserDefinedFunction =
    udf { annotatorProperties: Seq[Row] => function(annotatorProperties.map(Annotation(_))) }

  implicit class MapAnnotations(dataset: DataFrame) {
    def mapAnnotationsCol[T: TypeTag](column: String, outputCol: String, function: Seq[Annotation] => T): DataFrame = {
      val meta = dataset.schema(column).metadata
      val func = udf {
        annotatorProperties: Seq[Row] =>
          function(annotatorProperties.map(Annotation(_)))
      }
      dataset.withColumn(outputCol, func(col(column)).as(outputCol, meta))
    }
  
    def mapAnnotationsCol[T: TypeTag](firstCol: String, secondCol:String, outputCol: String, function: (Seq[Annotation], Seq[Annotation]) => T): DataFrame = {
      val firstMeta = dataset.schema(firstCol).metadata
      val secondMeta = dataset.schema(secondCol).metadata

      // TODO wanna impose some constraints? do it here :)
      val mergedMetadata = firstMeta

      val func = udf {
        (firstColAnnotations: Seq[Row], secondColAnnotations: Seq[Row]) =>
          function(firstColAnnotations.map(Annotation(_)), secondColAnnotations.map(Annotation(_)))
      }
      dataset.withColumn(outputCol, func(col(firstCol), col(secondCol)).as(outputCol, mergedMetadata))
    }


  }

  implicit class EachAnnotations(dataset: DataFrame) {

    import dataset.sparkSession.implicits._

    def eachAnnotationsCol[T: TypeTag](column: String, function: Seq[Annotation] => Unit): Unit = {
      dataset.select(column).as[Array[Annotation]].foreach(function(_))
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
