package com.johnsnowlabs.nlp.eval

import com.johnsnowlabs.nlp.annotator.PerceptronModel
import org.apache.spark.sql.SparkSession
import org.scalatest.FlatSpec

class POSEvalTestSpec extends FlatSpec {

  private val spark = SparkSession.builder()
    .appName("benchmark")
    .master("local[*]")
    .config("spark.driver.memory", "8G")
    .config("spark.kryoserializer.buffer.max", "200M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  "A POS Evaluation with IOB tags" should "display accuracy results for pretrained model" in {

    val testFile = "./eng.testb"
    val nerModel = PerceptronModel.pretrained()

    val posEvaluation = new POSEvaluation(spark, testFile)
    posEvaluation.computeAccuracyModel(nerModel)

  }

}
