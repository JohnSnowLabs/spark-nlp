package com.jsl.nlp

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, Row}

/**
  * Represents raw text container
  * @param text Raw text to be a target of processing
  * @param id May be used as an identifier for any specific needs
  * @param metadata May be used to provide additional useful information
  */
case class Document(
                     text: String,
                     id: String = "",
                     metadata: scala.collection.Map[String, String] = Map()
)
object Document {

  /** Creates a document out of a [[Row]] or column [[StructType]]*/
  def apply(row: Row): Document = {
    Document(
      row.getString(0),
      row.getString(1),
      row.getMap[String, String](2)
    )
  }

  /**Waiting for empty map fix by Spark team*/
  def column(column: Column)(implicit idColumn: Option[Column] = None,
                             metadata: Option[Column] = None): Column = {
    struct(
      column,
      idColumn.getOrElse(expr(column.toString()).as("id")),
      metadata.getOrElse(lit(null).cast(MapType(StringType, StringType)).as("metadata"))
    )
  }

  /** Spark type representation shape of Document*/
  val dataType: StructType = StructType(Array(
    StructField("text", StringType),
    StructField("id", StringType),
    StructField("metadata", MapType(StringType, StringType))
  ))

}