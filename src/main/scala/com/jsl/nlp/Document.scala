package com.jsl.nlp

import org.apache.spark.sql.functions.{expr, map, struct}
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

  def column(column: Column)(implicit idColumn: Column = expr(column.toString()), metadata: Column = map()): Column = {
    struct(column, idColumn, metadata)
  }

  /** Spark type representation shape of Document*/
  val DocumentDataType: StructType = StructType(Array(
    StructField("text",StringType,nullable = true),
    StructField("id",StringType,nullable = true),
    StructField("metadata",MapType(StringType,StringType,valueContainsNull = true),nullable = true)
  ))

}