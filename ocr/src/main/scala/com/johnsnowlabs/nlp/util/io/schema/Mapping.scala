package com.johnsnowlabs.nlp.util.io.schema

import org.apache.spark.sql.Row

case class Mapping(
                    c: String,
                    p: Int,
                    x: Float,
                    y: Float,
                    width: Float,
                    height: Float
                  ) {
  override def toString: String = c
}

object Mapping {
  def fromRow(row: Row): Mapping = {
    Mapping(
      row.getString(0),
      row.getInt(1),
      row.getFloat(2),
      row.getFloat(3),
      row.getFloat(4),
      row.getFloat(5)
    )
  }
}
