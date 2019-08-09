package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by saif on 02/05/17.
  */
object DataBuilder extends FlatSpec with BeforeAndAfterAll { this: Suite =>

  import SparkAccessor.spark.implicits._

  def basicDataBuild(content: String*)(implicit cleanupMode: String = "disabled"): Dataset[Row] = {
    val data = SparkAccessor.spark.sparkContext.parallelize(content).toDS().toDF("text")
    AnnotatorBuilder.withDocumentAssembler(data, cleanupMode)
  }

  def multipleDataBuild(content: Seq[String]): Dataset[Row] = {
    val data = SparkAccessor.spark.sparkContext.parallelize(content).toDS().toDF("text")
    AnnotatorBuilder.withDocumentAssembler(data)
  }

  def buildNerDataset(datasetContent: String): Dataset[Row] = {
    val lines = datasetContent.split("\n")
    val data = CoNLL(conllLabelIndex = 1)
      .readDatasetFromLines(lines, SparkAccessor.spark).toDF
    AnnotatorBuilder.withDocumentAssembler(data)
  }

  def loadParquetDataset(path: String) =
    SparkAccessor.spark.read.parquet(path)
}
