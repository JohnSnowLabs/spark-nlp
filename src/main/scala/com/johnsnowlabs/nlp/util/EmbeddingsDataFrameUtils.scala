package com.johnsnowlabs.nlp.util

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

object EmbeddingsDataFrameUtils {
  // Schema Spark expects for `format("image")`
  val imageSchema: StructType = StructType(
    Seq(
      StructField(
        "image",
        StructType(Seq(
          StructField("origin", StringType, true),
          StructField("height", IntegerType, true),
          StructField("width", IntegerType, true),
          StructField("nChannels", IntegerType, true),
          StructField("mode", IntegerType, true),
          StructField("data", BinaryType, true))))))

  // A reusable null image row for text-only embedding scenarios
  val emptyImageRow: Row = Row(Row("", 0, 0, 0, 0, Array[Byte]()))
}
