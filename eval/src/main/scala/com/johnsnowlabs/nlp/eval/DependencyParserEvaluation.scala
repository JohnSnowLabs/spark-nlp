package com.johnsnowlabs.nlp.eval

import scala.collection.mutable

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._

import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.util.{Benchmark, PipelineModels}

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions._

object DependencyParserEvaluation extends App {

  private val spark = SparkSession.builder()
    .appName("benchmark")
    .master("local[*]")
    .config("spark.driver.memory", "2G")
    .config("spark.kryoserializer.buffer.max", "200M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  import spark.implicits._

  private val testingFile = "src/main/resources/parsers/test.conllu.txt"
  private val trainingFile = "src/main/resources/parsers/train.conllu.small.txt"

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val sentenceDetector = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

  private val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

  private val posTagger = PerceptronModel.pretrained()
    .setInputCols(Array("token", "sentence"))
    .setOutputCol("pos")

  private val dependencyParserConllU = new DependencyParserApproach()
    .setInputCols(Array("sentence", "pos", "token"))
    .setOutputCol("dependency")
    .setConllU(trainingFile)
    .setNumberOfIterations(15)

  private val pipeline = new Pipeline()
    .setStages(Array(
      documentAssembler,
      sentenceDetector,
      tokenizer,
      posTagger,
      dependencyParserConllU
    ))

  private val emptyDataSet = PipelineModels.dummyDataset
  var dependencyParserModel: PipelineModel = _
  Benchmark.measure("[Dependency Parser] Time to train") {
    dependencyParserModel = pipeline.fit(emptyDataSet)
  }

  val testDataSet = getTestDataSet(testingFile)
  println("Test Dataset")
  testDataSet.show(100,false)

  val predictionDataSet = dependencyParserModel.transform(testDataSet)
  Benchmark.measure("[Dependency Parser] Time to show") {
    predictionDataSet.select("dependency").show(false)
  }

  println("Ground Truth Dataset")
  private val groundTruthDataSet = getCoNLLUDataSet(testingFile).select("form", "head")
  groundTruthDataSet.show(100,false)

  println("Evaluation Dataset")
  private val evaluationDataSet = getEvaluationDataSet(predictionDataSet, groundTruthDataSet)
  evaluationDataSet.show(100, false)
  computeAccuracy(evaluationDataSet)

  def getTestDataSet(testFile: String): Dataset[_] = {
    val testDataSet = getCoNLLUDataSet(testFile).select("id", "form")
    testDataSet.withColumn("sentence_id", getSentenceId($"id"))
               .groupBy("sentence_id")
               .agg(count("id"), collect_list("form").as("form_list"))
               .withColumn("text", concat_ws(" " , $"form_list"))
               .select("text")
               .sort($"sentence_id")
  }

  private var counter = 0

  private def getSentenceId = udf { tokenId: String =>
    if (tokenId == "1") {
      counter += 1
    }
    counter.toString
  }

  private def getDependencies(relations: Seq[String]): Seq[String] = {
    relations.map(relation => relation.split(",")(1)
                                      .replace(")",""))
  }

  private def getHeads(metaData: Seq[Map[String, String]]): Seq[String] = {
    metaData.map(metaDatum => metaDatum.getOrElse("head", "-1"))
  }

  def getCoNLLUDataSet(conllUFile: String): Dataset[_]  = {
    spark.read.option("comment", "#").option("delimiter", "\\t")
      .csv(conllUFile)
      .withColumnRenamed("_c0", "id")
      .withColumnRenamed("_c1", "form")
      .withColumnRenamed("_c2", "lemma")
      .withColumnRenamed("_c3", "upos")
      .withColumnRenamed("_c4", "xpos")
      .withColumnRenamed("_c5", "feats")
      .withColumnRenamed("_c6", "head")
      .withColumnRenamed("_c7", "deprel")
      .withColumnRenamed("_c8", "deps")
      .withColumnRenamed("_c9", "misc")
  }

  def getEvaluationDataSet(predictionDataSet: Dataset[_], groundTruthDataSet: Dataset[_]): Dataset[_] = {
    val cleanPredictionDataSet = processPredictionDataSet(predictionDataSet)
    println("Prediction Dataset")
    cleanPredictionDataSet.show(100, false)
    val evaluationDataSet = cleanPredictionDataSet.withColumn("id", monotonically_increasing_id())
      .join(groundTruthDataSet.withColumn("id", monotonically_increasing_id()), Seq("id"))
      .drop("id")
    evaluationDataSet.select("form", "predicted_head", "head")
  }

  def processPredictionDataSet(dataSet: Dataset[_]): Dataset[_] = {
    val cleanPrediction: Seq[(String, String)] =  predictionDataSet.select("dependency.result",
      "dependency.metadata").rdd.map { row =>
      val relations: Seq[String] = row.get(0).asInstanceOf[mutable.WrappedArray[String]].toList
      val dependencies = getDependencies(relations)
      val metaData: Seq[Map[String, String]] = row.get(1).asInstanceOf[mutable.WrappedArray[Map[String, String]]]
      val heads = getHeads(metaData)
      (dependencies, heads)
    }.collect().flatMap(row => row._1 zip row._2)
    cleanPrediction.toDF("form_prediction", "predicted_head")
  }

  def computeAccuracy(evaluationDataSet: Dataset[_]): Unit = {
    val accuracyDataSet = evaluationDataSet.withColumn("result",
      when($"predicted_head" === $"head", 1).otherwise(0))
    accuracyDataSet.select(avg(col("result"))).alias("accuracy").show()
  }

}
