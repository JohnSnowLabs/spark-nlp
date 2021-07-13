package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.nlp.Annotation
import org.apache.spark.sql.types.{ArrayType, MapType, StringType, StructField, StructType}

object FinisherUtil {

  def checkIfInputColsExist(inputCols: Array[String], schema: StructType): Unit = {
    require(inputCols.forall(schema.fieldNames.contains),
      s"pipeline annotator stages incomplete. " +
        s"expected: ${inputCols.mkString(", ")}, " +
        s"found: ${schema.fields.filter(_.dataType == ArrayType(Annotation.dataType)).map(_.name).mkString(", ")}, " +
        s"among available: ${schema.fieldNames.mkString(", ")}")
  }

  def checkIfAnnotationColumnIsSparkNLPAnnotation(schema: StructType, annotationColumn: String): Unit = {
    require(schema(annotationColumn).dataType == ArrayType(Annotation.dataType),
      s"column [$annotationColumn] must be an NLP Annotation column")
  }

  def getMetadataFields(outputCols: Array[String], outputAsArray: Boolean): Array[StructField] = {
    outputCols.flatMap(outputCol => {
      if (outputAsArray)
        Some(StructField(outputCol + "_metadata", MapType(StringType, StringType), nullable = false))
      else
        None
    })
  }

  def getOutputFields(outputCols: Array[String], outputAsArray: Boolean): Array[StructField] = {
    outputCols.map(outputCol => {
      if (outputAsArray)
        StructField(outputCol, ArrayType(StringType), nullable = false)
      else
        StructField(outputCol, StringType, nullable = true)
    })
  }

  def getCleanFields(cleanAnnotations: Boolean, outputFields: Array[StructField]): Array[StructField] = {
    if (cleanAnnotations) outputFields.filterNot(f =>
      f.dataType == ArrayType(Annotation.dataType)
    ) else outputFields
  }

}
