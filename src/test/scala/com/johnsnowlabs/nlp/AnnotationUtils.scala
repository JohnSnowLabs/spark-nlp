package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import org.apache.spark.sql.types.{MetadataBuilder, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}

object AnnotationUtils {

  private lazy val spark = SparkAccessor.spark

  implicit class AnnotationRow(annotation: Annotation) {

    def toRow(): Row = {
      Row(
        annotation.annotatorType,
        annotation.begin,
        annotation.end,
        annotation.result,
        annotation.metadata,
        annotation.embeddings)
    }
  }

  implicit class DocumentRow(s: String) {
    def toRow(metadata: Map[String, String] = Map("sentence" -> "0")): Row = {
      Row(Seq(Annotation(DOCUMENT, 0, s.length, s, metadata).toRow()))
    }
  }

  /** Create a DataFrame with the given column name, annotator type and annotations row Output
    * column will be compatible with the Spark NLP annotators
    */
  def createAnnotatorDataframe(
      columnName: String,
      annotatorType: String,
      annotationsRow: Row): DataFrame = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", annotatorType)
    val documentField =
      StructField(columnName, Annotation.arrayType, nullable = false, metadataBuilder.build)
    val struct = StructType(Array(documentField))
    val rdd = spark.sparkContext.parallelize(Seq(annotationsRow))
    spark.createDataFrame(rdd, struct)
  }

}
