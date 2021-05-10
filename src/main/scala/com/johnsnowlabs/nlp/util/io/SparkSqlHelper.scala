package com.johnsnowlabs.nlp.util.io

import com.johnsnowlabs.util.Version
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{array_distinct, col, flatten, udf}
import org.apache.spark.sql.types.ArrayType

object SparkSqlHelper {

  lazy val sparkVersion: Float = {
    val version = Version.parse(ResourceHelper.spark.version)
    val versionParts = version.parts.map( v => v.toString)
    versionParts.reduceLeft((x, y) => x + "." + y).slice(0 ,1).toFloat
  }

  def uniqueArrayElements(dataset: Dataset[_], column: String): Dataset[_] = {
    val columnName = "unique_" + column + "_elements"
    dataset.schema(column).dataType match {
      case ArrayType(_, _) => dataset.schema(column).dataType.asInstanceOf[ArrayType].elementType.typeName match {
        case "array" => getUniqueElementsFromNestedArray(dataset, column, columnName)
        case _ => getUniqueElements(dataset, column, columnName)
      }
      case _ => dataset
    }
  }

  private def getUniqueElements(dataset: Dataset[_], column: String, uniqueColumnName: String): Dataset[_] = {
    if (sparkVersion > 2.3) {
      dataset.withColumn(uniqueColumnName, array_distinct(col(column)))
    } else {
      dataset.withColumn(uniqueColumnName, removeDuplicatedElements(col(column)))
    }
  }

  private def getUniqueElementsFromNestedArray(dataset: Dataset[_], column: String, uniqueColumnName: String):
  Dataset[_] = {
    if (sparkVersion > 2.3) {
      dataset.withColumn(uniqueColumnName, array_distinct(flatten(col(column))))
    } else {
      dataset.withColumn(uniqueColumnName, removeDuplicatedNestedElements(col(column)))
    }
  }

  private def removeDuplicatedElements: UserDefinedFunction = udf { (column: Seq[String]) =>
    column.distinct
  }

  private def removeDuplicatedNestedElements: UserDefinedFunction = udf { (column: Seq[Seq[String]]) =>
    column.flatMap(c => c.distinct).distinct
  }

}
