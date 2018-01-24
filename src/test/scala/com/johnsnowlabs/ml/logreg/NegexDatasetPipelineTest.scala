package com.johnsnowlabs.ml.logreg

import com.johnsnowlabs.ml.common.EvaluationMetrics
import com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegApproach
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object NegexDatasetPipelineTest extends App with EvaluationMetrics {

  implicit val spark = SparkSession.builder().appName("i2b2 logreg").master("local[1]").getOrCreate
  import spark.implicits._

  // directory of the i2b2 dataset
  val i2b2Dir = "/home/jose/Downloads/i2b2"

  // word embeddings location
  val embeddingsFile = s"/home/jose/Downloads/bio_nlp_vec/PubMed-shuffle-win-2.bin"
  val datasetPath = "rsAnnotations-1-120-random.txt"
  val embeddingsDims = 200

  val reader = new NegexDatasetReader()

  val dataset = "rsAnnotations-1-120-random.txt"

  val ds = reader.readDataframe(datasetPath).cache

  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = ds.randomSplit(Array(0.7, 0.3))

  val model = trainAssertionModel(trainingData.cache)
  var result = testAssertionModel(testData.cache, model)

  var pred = result.select($"assertion").collect.map(row => Annotation(row.getAs[Seq[Row]]("assertion").head).result)
  var gold = result.select($"label").collect.map(_.getAs[String]("label"))

  println(calcStat(pred, gold))
  println(confusionMatrix(pred, gold))

  /* test serialization */
  val modelName = "assertion_model"
  model.write.overwrite().save(modelName)
  val readModel = PipelineModel.read.load(modelName)

  result = testAssertionModel(testData, readModel)
  pred = result.select($"assertion").collect.map(row => Annotation(row.getAs[Seq[Row]]("assertion").head).result)
  gold = result.select($"label").collect.map(_.getAs[String]("label"))

  println(calcStat(pred, gold))
  println(confusionMatrix(pred, gold))

  def getAssertionStages(): Array[_ <: PipelineStage] = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("sentence")
      .setOutputCol("document")

    val assertionStatus = new AssertionLogRegApproach()
      .setLabelCol("label")
      .setInputCols("document")
      .setOutputCol("assertion")
      .setBefore(11)
      .setAfter(13)
      .setEmbeddingsSource(embeddingsFile, 200, WordEmbeddingsFormat.Binary)

    Array(documentAssembler,
      assertionStatus)
  }

  def trainAssertionModel(dataset: DataFrame): PipelineModel = {

    System.out.println("Start fitting")

    // train Assertion Status
    val pipeline = new Pipeline()
      .setStages(getAssertionStages)

    pipeline.fit(dataset)
  }

  def testAssertionModel(dataset: DataFrame, model: PipelineModel) = {
    System.out.println("Test Dataset Reading")
    model.transform(dataset)
  }

}
