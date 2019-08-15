package com.johnsnowlabs.nlp.annotators.ocr.schema

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

case class PageMatrix(mapping: Array[Mapping]) {
  override def toString: String = mapping.map(_.toString).mkString
}

object PageMatrix {

  val dataType =
    ArrayType(StructType(Seq(
      StructField("mapping", ArrayType(Mapping.mappingType, true),true)
    )), true)

  def fromRow(row: Row) = PageMatrix(
    row.getSeq[Row](0).map(Mapping.fromRow).toArray
  )

}
