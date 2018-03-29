package com.johnsnowlabs.ml.lstm

import com.johnsnowlabs.ml.common.EvaluationMetrics
import com.johnsnowlabs.ml.logreg.NegexDatasetReader
import com.johnsnowlabs.nlp.annotators.assertion.dl.AssertionDLApproach
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/*
* Test Assertion Status on the Pipeline
* dataset from NegEx, can be obtained from,
* https://raw.githubusercontent.com/mongoose54/negex/master/genConText/rsAnnotations-1-120-random.txt
* Word Embeddings can be obtained from,
* https://github.com/cambridgeltl/BioNLP-2016
* */

object AssertionPipelineDLNegex extends App with EvaluationMetrics {

  implicit val spark = SparkSession.builder().appName("negex lstm").master("local[1]").getOrCreate
  import spark.implicits._

  // word embeddings location
  val embeddingsFile = s"PubMed-shuffle-win-2.bin"
  val datasetPath = "rsAnnotations-1-120-random.txt"
  val embeddingsDims = 200

  val reader = new NegexDatasetReader()

  val dataset = "rsAnnotations-1-120-random.txt"

  val ds = reader.readDataframe(datasetPath).cache

  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testingData) = ds.randomSplit(Array(0.7, 0.3))

  val model = trainAssertionModel(trainingData.cache)
  var result = testAssertionModel(testingData.cache, model)

  var pred = result.select($"assertion").collect.map(row => Annotation(row.getAs[Seq[Row]]("assertion").head).result)
  var gold = result.select($"label").collect.map(_.getAs[String]("label"))

  println(calcStat(pred, gold))
  println(confusionMatrix(pred, gold))

  /* test serialization */
  val modelName = "assertion_model"
  model.write.overwrite().save(modelName)
  val readModel = PipelineModel.read.load(modelName)

  result = testAssertionModel(testingData, readModel)
  pred = result.select($"assertion").collect.map(row => Annotation(row.getAs[Seq[Row]]("assertion").head).result)
  gold = result.select($"label").collect.map(_.getAs[String]("label"))

  println(calcStat(pred, gold))
  println(confusionMatrix(pred, gold))

  def getAssertionStages(): Array[_ <: PipelineStage] = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("sentence")
      .setOutputCol("document")

    val assertionStatus = new AssertionDLApproach()
      .setLabelCol("label")
      .setInputCols("document")
      .setOutputCol("assertion")
      .setBatchSize(16)
      .setEpochs(5)
      .setEmbeddingsSource(embeddingsFile, 200, WordEmbeddingsFormat.BINARY)

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
