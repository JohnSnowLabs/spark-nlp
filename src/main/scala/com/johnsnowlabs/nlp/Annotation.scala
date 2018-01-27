package com.johnsnowlabs.nlp

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.udf

import scala.collection.Map

/**
  * represents annotator's output parts and their details
  * @param annotatorType the type of annotation
  * @param start the index of the first character under this annotation
  * @param end the index after the last character under this annotation
  * @param metadata associated metadata for this annotation
  */
case class Annotation(annotatorType: String, start: Int, end: Int, result: String, metadata: Map[String, String])

object Annotation {

  private case class AnnotationContainer(__annotation: Array[Annotation])

  object extractors {
    /** annotation container ready for extraction */
    protected class AnnotationData(dataset: Dataset[Row]){
      def collect(column: String): Array[Array[Annotation]] = {
        Annotation.collect(dataset, column)
      }
      def take(column: String, howMany: Int): Array[Array[Annotation]] = {
        Annotation.take(dataset, column, howMany)
      }
    }
    implicit def data2andata(dataset: Dataset[Row]): AnnotationData = new AnnotationData(dataset)
  }

  private val ANNOTATION_NAME = "__annotation"
  val RESULT = "result"

  /** This is spark type of an annotation representing its metadata shape */
  val dataType = new StructType(Array(
    StructField("annotatorType", StringType, nullable = true),
    StructField("begin", IntegerType, nullable = false),
    StructField("end", IntegerType, nullable = false),
    StructField("result", StringType, nullable = true),
    StructField("metadata", MapType(StringType, StringType), nullable = true)
  ))


  /**
    * This method converts a [[org.apache.spark.sql.Row]] into an [[Annotation]]
    * @param row spark row to be converted
    * @return annotation
    */
  def apply(row: Row): Annotation = {
    Annotation(
      row.getString(0),
      row.getInt(1),
      row.getInt(2),
      row.getString(3),
      row.getMap[String, String](4)
    )
  }
  def apply(rawText: String): Annotation = Annotation(
    AnnotatorType.DOCUMENT,
    0,
    rawText.length,
    rawText,
    Map.empty[String, String]
  )

  /** dataframe collect of a specific annotation column*/
  def collect(dataset: Dataset[Row], column: String): Array[Array[Annotation]] = {
    require(dataset.columns.contains(column), s"column $column not present in data")
    import dataset.sparkSession.implicits._
    dataset
      .withColumnRenamed(column, ANNOTATION_NAME)
      .select(ANNOTATION_NAME)
      .as[AnnotationContainer]
      .map(_.__annotation)
      .collect
  }

  /** dataframe take of a specific annotation column */
  def take(dataset: Dataset[Row], column: String, howMany: Int): Array[Array[Annotation]] = {
    require(dataset.columns.contains(column), s"column $column not present in data")
    import dataset.sparkSession.implicits._
    dataset
      .withColumnRenamed(column, ANNOTATION_NAME)
      .select(ANNOTATION_NAME)
      .as[AnnotationContainer]
      .map(_.__annotation)
      .take(howMany)
  }

  /** dataframe annotation flatmap of metadata values */
  def flatten(vSep: String, aSep: String): UserDefinedFunction = {
    udf {
      (annotations: Seq[Row]) => annotations.map(r =>
        r.getString(3)
      ).mkString(aSep)
    }
  }

  /** dataframe annotation flatmap of metadata key values */
  def flattenKV(vSep: String, aSep: String): UserDefinedFunction = {
    udf {
      (annotations: Seq[Row]) => annotations.map(r =>
        (r.getMap[String, String](4) ++ Map(RESULT -> r.getString(3))).mkString(vSep).replace(" -> ", "->")
      ).mkString(aSep)
    }
  }

  /** dataframe annotation flatmap of metadata values as ArrayType */
  def flattenArray: UserDefinedFunction = {
    udf {
      (annotations: Seq[Row]) => annotations.map(r =>
        r.getString(3)
      )
    }
  }


}