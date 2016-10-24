package sparknlp

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

case class Annotation(aType: String, begin: Int, end: Int, metadata: scala.collection.Map[String, String] = Map())

object Annotation {
  def apply(row: Row): Annotation = {
    Annotation(row.getString(0), row.getInt(1), row.getInt(2), row.getMap[String, String](3))
  }

  val AnnotationDataType = new StructType(Array(
    StructField("aType", StringType, nullable = true),
    StructField("begin", IntegerType, nullable = true),
    StructField("end", IntegerType, nullable = true),
    StructField("metadata", MapType(StringType, StringType, valueContainsNull = true), nullable = true)
  ))
}