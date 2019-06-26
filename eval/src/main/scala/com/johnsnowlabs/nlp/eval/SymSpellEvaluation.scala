package com.johnsnowlabs.nlp.eval

import java.io.File
import scala.collection.mutable

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._

import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.util.{Benchmark, PipelineModels}

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions._

object SymSpellEvaluation extends App {

  private val spark = SparkSession.builder()
    .appName("benchmark")
    .master("local[1]")
    .config("spark.driver.memory", "8G")
    .config("spark.kryoserializer.buffer.max", "200M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  import spark.implicits._

  private val trainingDataSet = getDataSetFromFile("src/main/resources/spell/coca2017.txt")

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  private val spell = new SymmetricDeleteApproach()
    .setInputCols(Array("token"))
    .setOutputCol("spell")
    .setDictionary("src/main/resources/spell/words.txt")

  private val finisher = new Finisher()
    .setInputCols("spell")
    .setOutputCols("prediction")

  private val spellCheckerPipeline = new Pipeline()
    .setStages(Array(
      documentAssembler,
      tokenizer,
      spell,
      finisher
    ))

  val testDataSet = getDataSetFromFile("./misspell.txt")
  var spellCheckerModel: PipelineModel = _
  Benchmark.measure("[Symmetric Spell Checker] Time to train") {
    spellCheckerModel = spellCheckerPipeline.fit(trainingDataSet)
  }
  println("Prediction DataSet")
  val predictionDataSet = spellCheckerModel.transform(testDataSet).select("prediction")
  Benchmark.measure("[Symmetric Spell Checker] Time to show") {
    predictionDataSet.show()
  }
  val groundTruthDataSet = getGroundTruthDataSet("./ground_truth.txt")

  println("Evaluation DataSet")
  val evaluationDataSet = getEvaluationDataSet(predictionDataSet, groundTruthDataSet)
  evaluationDataSet.show(5, false)
  evaluationDataSet.select(avg(col("accuracy"))).show()

  def getDataSetFromFile(textFile: String): Dataset[_] = {
    if (textFile == "") {
      Seq("Simple data set").toDF.withColumnRenamed("value", "text")
    } else {
      spark.read.textFile(textFile)
        .withColumnRenamed("value", "text")
        .filter(row => !(row.mkString("").isEmpty && row.length > 0))
    }
  }

  def getGroundTruthDataSet(textFile: String): Dataset[_] = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val finisher = new Finisher()
      .setInputCols("token")
      .setOutputCols("ground_truth")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        finisher
      ))

    val groundTruthDataSet = getDataSetFromFile(textFile)

    pipeline.fit(PipelineModels.dummyDataset)
      .transform(groundTruthDataSet)
      .select("ground_truth")

  }

  def getEvaluationDataSet(predictionDataSet: Dataset[_], groundTruthDataSet: Dataset[_]): Dataset[_] = {
    val evaluationDataSet = predictionDataSet.withColumn("id", monotonically_increasing_id())
      .join(groundTruthDataSet.withColumn("id", monotonically_increasing_id()), Seq("id"))
      .drop("id")
    evaluationDataSet.withColumn("accuracy",
      getAccuracy(col("prediction"), col("ground_truth")))
  }

  private def getAccuracy = udf { (prediction: mutable.WrappedArray[String],
                                   groundTruth: mutable.WrappedArray[String]) =>
    val accuracy = computeAccuracy(prediction.toSet, groundTruth.toSet)
    accuracy
  }

  def computeAccuracy(prediction: Set[String], groundTruth: Set[String]): Float = {
    val numberOfCorrectWords = prediction.intersect(groundTruth).size.toFloat
    val accuracy: Float = numberOfCorrectWords / groundTruth.size.toFloat
    accuracy
  }

}
