package com.johnsnowlabs.util.spark

import org.apache.spark.sql.Dataset

object SparkUtil {

  def retrieveColumnName(dataset: Dataset[_], annotatorType: String): String = {
    val structFields = dataset.schema.fields
      .filter(field => field.metadata.contains("annotatorType"))
      .filter(field => field.metadata.getString("annotatorType") == annotatorType)
    val columnNames = structFields.map(structField => structField.name)

    columnNames.head
  }

}
