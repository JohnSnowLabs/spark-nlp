package sparknlp

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{MapType, StringType, StructField, StructType}

case class Document(
  id: String,
  text: String,
  metadata: scala.collection.Map[String, String] = Map()
)

object Document {
  def apply(row: Row): Document = {
    Document(row.getString(0), row.getString(1), row.getMap[String, String](2))
  }

  val DocumentDataType = StructType(Array(
    StructField("id",StringType,nullable = true),
    StructField("text",StringType,nullable = true),
    StructField("metadata",MapType(StringType,StringType,valueContainsNull = true),nullable = true)
  ))
}