package com.johnsnowlabs.ml.lstm.PersistI2b2

import com.johnsnowlabs.ml.logreg.I2b2DatasetReader
import org.apache.spark.sql.SparkSession

/**
  * Created by jose on 26/02/18.
  *
  * Silly utility class to persist dataset to CSV
  */
object PersistI2b2 extends App {

  implicit val spark = SparkSession.builder().appName("i2b2 logreg").master("local[1]").getOrCreate
  import spark.implicits._

  // directory of the i2b2 dataset
  val i2b2Dir = "/home/jose/Downloads/i2b2"

  val trainDatasetPath = Seq(s"${i2b2Dir}/concept_assertion_relation_training_data/partners",
    s"${i2b2Dir}/concept_assertion_relation_training_data/beth")
  val reader = new I2b2DatasetReader(wordEmbeddingsFile = null, targetLengthLimit = 12)
  val trainAnnotations = reader.read(trainDatasetPath)
  trainAnnotations.toDF.write.option("header", true).csv("i2b2.csv")

}
