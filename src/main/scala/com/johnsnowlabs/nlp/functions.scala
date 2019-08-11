package com.johnsnowlabs.nlp

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.{array, col, explode, udf}
import org.apache.spark.sql.types._

import scala.reflect.runtime.universe._

object functions {

  implicit class FilterAnnotations(dataset: DataFrame) {
    def filterByAnnotations(column: String, function: Seq[Annotation] => Boolean): DataFrame = {
      val meta = dataset.schema(column).metadata
      val func = udf {
        annotatorProperties: Seq[Row] =>
          function(annotatorProperties.map(Annotation(_)))
      }
      dataset.filter(func(col(column)).as(column, meta))
    }
  }

  implicit class MapAnnotations(dataset: DataFrame) {
    def mapAnnotations[T: TypeTag](column: String, outputCol: String, function: Seq[Annotation] => T): DataFrame = {
      val meta = dataset.schema(column).metadata
      val func = udf {
        annotatorProperties: Seq[Row] =>
          function(annotatorProperties.map(Annotation(_)))
      }
      dataset.withColumn(outputCol, func(col(column)).as(outputCol, meta))
    }
  }

  implicit class EachAnnotations(dataset: DataFrame) {

    import dataset.sparkSession.implicits._

    def eachAnnotations[T: TypeTag](column: String, function: Seq[Annotation] => Unit): Unit = {
      dataset.select(column).as[Array[Annotation]].foreach(function(_))
    }
  }

  implicit class ExplodeAnnotations(dataset: DataFrame) {
    def explodeAnnotations[T: TypeTag](column: String, outputCol: String): DataFrame = {
      val meta = dataset.schema(column).metadata
      dataset.
        withColumn(outputCol, explode(col(column))).
        withColumn(outputCol, array(col(outputCol)).as(outputCol, meta))
    }
  }

  case class CoverageResult(covered: Long, total: Long, percentage: Float) extends Serializable

  implicit class EmbeddingsCoverage(dataset: DataFrame) {

    def withCoverageColumn[T: TypeTag](embeddingsColumn: String, outputCol: String): DataFrame = {
      val coverageFn = udf((annotatorProperties: Seq[Row]) => {
        val annotations = annotatorProperties.map(Annotation(_))
        val oov = annotations.map(x => if (x.metadata("isOOV") == "false") 1 else 0)
        val covered = oov.sum
        val total = annotations.count(_ => true)
        val percentage = 1f * covered / total
        CoverageResult(covered, total, percentage)
      })
      dataset.withColumn(outputCol, coverageFn(col(embeddingsColumn)))
    }


    def overallCoverage[T: TypeTag](embeddingsColumn: String): CoverageResult = {
      val words = dataset.select(embeddingsColumn).rdd.flatMap(row => {
        val annotations = row.getAs[Seq[Row]](embeddingsColumn)
        annotations.map(annotation => Tuple2(
          annotation.getAs[Map[String, String]]("metadata")("token"),
          if (annotation.getAs[Map[String, String]]("metadata")("isOOV") == "false") 1 else 0))
      })
      val oov_sum = words.reduce((a, b) => Tuple2("Total", a._2 + b._2))
      val covered = oov_sum._2
      val total = words.count()
      val percentage = 1f * covered / total
      CoverageResult(covered, total, percentage)
    }
  }

}
