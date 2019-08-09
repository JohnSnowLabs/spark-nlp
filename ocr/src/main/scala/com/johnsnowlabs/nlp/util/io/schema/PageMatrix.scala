package com.johnsnowlabs.nlp.util.io.schema

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

case class PageMatrix(
                       mapping: Array[Mapping],
                       lowerLeftX: Float,
                       lowerLeftY: Float
                     ) {
  override def toString: String = mapping.map(_.toString).mkString
}

object PageMatrix {

  private val coordinatesType =
    StructType(Seq(
      StructField("c", StringType, true),
      StructField("x",FloatType, false),
      StructField("y",FloatType, false),
      StructField("width",FloatType, false),
      StructField("height",FloatType, false)
    ))

  val dataType =
    StructType(Seq(
      StructField("mapping", ArrayType(coordinatesType),true),
      StructField("lowerLeftX", FloatType,false),
      StructField("lowerLeftY", FloatType,false)
    ))

  def fromRow(row: Row) = PageMatrix(
    row.getSeq[Row](0).map(Mapping.fromRow).toArray,
    row.getFloat(1),
    row.getFloat(2)
  )

}
