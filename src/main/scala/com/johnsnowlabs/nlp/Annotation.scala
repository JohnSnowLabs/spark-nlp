package com.johnsnowlabs.nlp

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.udf

import scala.collection.Map

/**
  * represents annotator's output parts and their details
  * @param annotatorType the type of annotation
  * @param result summary of annotations output
  * @param metadata associated metadata for this annotation
  */
case class Annotation(annotatorType: String, result: String, metadata: Map[String, String])

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
  val BEGIN = "begin"
  val END = "end"
  val RESULT = "result"

  /** This is spark type of an annotation representing its metadata shape */
  val dataType = new StructType(Array(
    StructField("annotatorType", StringType, nullable = false),
    StructField("begin", IntegerType, nullable = false),
    StructField("end", IntegerType, nullable = false),
    StructField("result", StringType, nullable = false),
    StructField("metadata", MapType(StringType, StringType), nullable = false)
  ))


  /**
    * This method converts a [[org.apache.spark.sql.Row]] into an [[Annotation]]
    * @param row spark row to be converted
    * @return annotation
    */
  def apply(row: Row): Annotation = {
    Annotation(
      row.getString(0),
      row.getString(1),
      row.getMap[String, String](2)
    )
  }
  def apply(rawText: String): Annotation = Annotation(
    AnnotatorType.DOCUMENT,
    rawText,
    Map(Annotation.BEGIN -> "0", Annotation.END -> (rawText.length - 1).toString)
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
        r.getString(1)
      ).mkString(aSep)
    }
  }

  /** dataframe annotation flatmap of metadata key values */
  def flattenKV(vSep: String, aSep: String): UserDefinedFunction = {
    udf {
      (annotations: Seq[Row]) => annotations.map(r =>
        (r.getMap[String, String](2) ++ Map(RESULT -> r.getString(1))).mkString(vSep).replace(" -> ", "->")
      ).mkString(aSep)
    }
  }


}