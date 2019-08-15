package com.johnsnowlabs.nlp.annotators.ocr.schema

import org.apache.spark.sql.types._

/**
  *
  * @param i  Chunk index.
  * @param p  Page number.
  * @param x  The lower left x coordinate.
  * @param y  The lower left y coordinate.
  * @param w  The width of the rectangle.
  * @param h  The height of the rectangle.
  */
case class Coordinate(i: Int, p: Int, x: Float, y: Float, w: Float, h: Float)
object Coordinate {
  val coordinateType = ArrayType(
    StructType(Seq(
      StructField("i",IntegerType,false),
      StructField("p",IntegerType,false),
      StructField("x",FloatType,false),
      StructField("y",FloatType,false),
      StructField("w",FloatType,false),
      StructField("h",FloatType,false)
    )), true
  )
}
