package com.johnsnowlabs.nlp.eval

import com.johnsnowlabs.nlp.annotator.{PerceptronApproach, PerceptronModel}
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

  val testFile = "./eng.testb"

  "A POS Evaluation" should "display accuracy results for a pre-trained model" in {

    val posModel = PerceptronModel.pretrained()

    val posEvaluation = new POSEvaluation(spark, testFile)
    posEvaluation.computeAccuracyModel(posModel)

  }

  "A POS Evaluation" should "display accuracy results" in {

    val trainFile = "src/test/resources/anc-pos-corpus-small/110CYL068.txt"

    val posTagger = new PerceptronApproach()
      .setInputCols(Array("document", "token"))
      .setOutputCol("pos")
      .setNIterations(2)

    val posEvaluation = new POSEvaluation(spark, testFile)
    posEvaluation.computeAccuracyAnnotator(trainFile, posTagger)

  }

}
