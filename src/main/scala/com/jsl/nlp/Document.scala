package com.jsl.nlp

import org.apache.spark.sql.types._
import org.apache.spark.sql.Row

case class Document(
  id: String,
  text: String,
  metadata: scala.collection.Map[String, String] = Map()
)

object Document extends ExtractsFromRow {
  def apply(row: Row): Document = {
    Document(
      row.getString(0),
      row.getString(1),
      row.getMap[String, String](2)
    )
  }

  val DocumentDataType: StructType = StructType(Array(
    StructField("id",StringType,nullable = true),
    StructField("text",StringType,nullable = true),
    StructField("metadata",MapType(StringType,StringType,valueContainsNull = true),nullable = true)
  ))

  override def validate(row: Row): Unit = ???

}