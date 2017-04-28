package com.jsl.nlp

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

/**
  * This class represents the pieces of text created identified by an annotator
  * @param aType the type of annotation
  * @param begin the index of the first character under this annotation
  * @param end the index after the last character under this annotation
  * @param metadata associated metadata for this annotation
  */
case class Annotation(aType: String, begin: Int, end: Int, metadata: scala.collection.Map[String, String] = Map())

object Annotation extends ExtractsFromRow {

  /**
    * This is spark type of an annotation
    */
  val AnnotationDataType = new StructType(Array(
    StructField("aType", StringType, nullable = true),
    StructField("begin", IntegerType, nullable = false),
    StructField("end", IntegerType, nullable = false),
    StructField("metadata", MapType(StringType, StringType, valueContainsNull = true), nullable = true)
  ))


  /**
    * This method converts a [[org.apache.spark.sql.Row]] into an [[Annotation]]
    * @param row the row to be converted
    * @return the annotation
    */
  def apply(row: Row): Annotation = {
    Annotation(row.getString(0), row.getInt(1), row.getInt(2), row.getMap[String, String](3))
  }

  override def validate(row: Row): Unit = ???

}