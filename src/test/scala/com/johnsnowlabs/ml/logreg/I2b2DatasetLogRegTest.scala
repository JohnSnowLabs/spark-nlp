package com.johnsnowlabs.ml.logreg

import org.apache.spark.sql.SparkSession

object I2b2DatasetLogRegTest extends App {

  implicit val spark = SparkSession.builder().appName("DataFrame-UDF").master("local[4]").getOrCreate()
  val datasetPath = "/home/jose/Downloads/concept_assertion_relation_training_data/beth"
  val reader = new I2b2DatasetReader(datasetPath)

  reader.readDataset.printSchema()

}
