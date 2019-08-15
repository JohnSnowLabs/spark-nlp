package com.johnsnowlabs.nlp.annotators.ocr.schema

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

case class PageMatrix(mapping: Array[Mapping]) {
  override def toString: String = mapping.map(_.toString).mkString
}

object PageMatrix {

  private val coordinatesType =
    StructType(Seq(
      StructField("c", StringType, true),
      StructField("p", IntegerType, true),
      StructField("x",FloatType, false),
      StructField("y",FloatType, false),
      StructField("width",FloatType, false),
      StructField("height",FloatType, false)
    ))

  val dataType =
    ArrayType(StructType(Seq(
      StructField("mapping", ArrayType(coordinatesType, true),true)
    )), true)

  def fromRow(row: Row) = PageMatrix(
    row.getSeq[Row](0).map(Mapping.fromRow).toArray
  )

}