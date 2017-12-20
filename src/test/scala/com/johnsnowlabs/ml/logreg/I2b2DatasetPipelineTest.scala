package com.johnsnowlabs.ml.logreg

import com.johnsnowlabs.ml.common.EvaluationMetrics
import com.johnsnowlabs.ml.logreg.I2b2DatasetLogRegTest.{calcStat, confusionMatrix}
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.RegexTokenizer
import com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegApproach
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

object I2b2DatasetPipelineTest extends App with EvaluationMetrics{

  implicit val spark = SparkSession.builder().appName("i2b2 logreg").master("local[4]")
        .config("spark.executor.memory", "2g").getOrCreate

  import spark.implicits._
  val trainPaths = Seq("/home/jose/Downloads/i2b2/concept_assertion_relation_training_data/partners"
  ,"/home/jose/Downloads/i2b2/concept_assertion_relation_training_data/beth")
  val testPaths = Seq("/home/jose/Downloads/i2b2/test_data")

  val embeddingsFile = s"/home/jose/Downloads/bio_nlp_vec/PubMed-shuffle-win-2.bin"

  def getAssertionStages(): Array[_ <: PipelineStage] = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new RegexTokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val assertionStatus = new AssertionLogRegApproach()
      .setInputCols("document")
      .setOutputCol("assertion")
      .setBefore(11)
      .setAfter(13)
      .setEmbeddingsSource(embeddingsFile, 200, WordEmbeddingsFormat.Binary)
      .setEmbeddingsFolder("/home/jose/Downloads/bio_nlp_vec")

    Array(documentAssembler,
      tokenizer,
      assertionStatus)
  }

  val reader = new I2b2DatasetReader(embeddingsFile)

  def trainAssertionModel(paths: Seq[String]): PipelineModel = {
    System.out.println("Train Dataset Reading")
    val time = System.nanoTime()
    val dataset = reader.readDataFrame(paths)
    System.out.println(s"Done, ${(System.nanoTime() - time)/1e9}\n")
    System.out.println("Start fitting")

    // train Assertion Status
    val pipeline = new Pipeline()
      .setStages(getAssertionStages)

    pipeline.fit(dataset.cache())
  }

  def testAssertionModel(path:Seq[String], model: PipelineModel) = {
    System.out.println("Test Dataset Reading")
    val dataset = reader.readDataFrame(path)
    model.transform(dataset.cache())
  }

  val model = trainAssertionModel(trainPaths)
  val result = testAssertionModel(testPaths, model)

  val pred = result.select($"prediction").collect.map(_.getAs[Double]("prediction"))
  val gold = result.select($"label").collect.map(_.getAs[Double]("label"))

  println(calcStat(pred, gold))
  println(confusionMatrix(pred, gold))
}
