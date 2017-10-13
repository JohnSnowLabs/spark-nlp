package com.johnsnowlabs.nlp

import com.johnsnowlabs.ml.crf.CoNLL
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by saif on 02/05/17.
  */
object DataBuilder extends FlatSpec with BeforeAndAfterAll { this: Suite =>

  import SparkAccessor.spark.implicits._

  def basicDataBuild(content: String*): Dataset[Row] = {
    val data = SparkAccessor.spark.sparkContext.parallelize(content).toDS().toDF("text")
    AnnotatorBuilder.withDocumentAssembler(data)
  }

  def multipleDataBuild(content: Seq[String]): Dataset[Row] = {
    val data = SparkAccessor.spark.sparkContext.parallelize(content).toDS().toDF("text")
    AnnotatorBuilder.withDocumentAssembler(data)
  }

  def buildNerDataset(datasetContent: String): Dataset[Row] = {
    val lines = datasetContent.split("\n")
    val data = new CoNLL(1, SparkAccessor.spark).readDatasetFromLines(lines).toDF
    AnnotatorBuilder.withDocumentAssembler(data)
  }
}
