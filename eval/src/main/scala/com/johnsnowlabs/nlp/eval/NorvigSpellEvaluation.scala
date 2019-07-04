package com.johnsnowlabs.nlp.eval

import scala.collection.mutable
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.eval.util.LoggingData
import com.johnsnowlabs.util.{Benchmark, PipelineModels}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions._

object NorvigSpellEvaluation extends App {

  var loggingData: LoggingData = _

  def apply(trainFile: String, spell: NorvigSweetingApproach, testFile: String, groundTruthFile: String): Unit = {
    loggingData = new LoggingData("LOCAL", this.getClass.getSimpleName)
    loggingData.logNorvigParams(spell)
    computeAccuracy(trainFile, spell, testFile, groundTruthFile)
    loggingData.closeLog()
  }

  private def computeAccuracy(trainFile: String, spell: NorvigSweetingApproach,
                              testFile: String, groundTruthFile: String): Unit = {
    val spellCheckerModel = trainSpellChecker(trainFile, spell)
    val predictionDataSet = correctMisspells(spellCheckerModel, testFile)
    evaluateSpellChecker(groundTruthFile, predictionDataSet)
  }

  private def trainSpellChecker(trainFile: String, spell: NorvigSweetingApproach): PipelineModel = {
    val trainingDataSet = getDataSetFromFile(trainFile)
    var spellCheckerModel: PipelineModel = null
    val spellCheckerPipeline = getSpellCheckerPipeline(spell)
    Benchmark.measure("[Norvig Spell Checker] Time to train") {
      spellCheckerModel = spellCheckerPipeline.fit(trainingDataSet)
    }
    spellCheckerModel
  }

  private def getDataSetFromFile(textFile: String): Dataset[_] = {

    val spark = SparkSession.builder()
      .appName("benchmark")
      .master("local[1]")
      .config("spark.kryoserializer.buffer.max", "200M")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()

    import spark.implicits._

    if (textFile == "") {
      Seq("Simple data set").toDF.withColumnRenamed("value", "text")
    } else {
      spark.read.textFile(textFile)
        .withColumnRenamed("value", "text")
        .filter(row => !(row.mkString("").isEmpty && row.length > 0))
    }
  }

  private def getSpellCheckerPipeline(spell: NorvigSweetingApproach): Pipeline =  {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val finisher = new Finisher()
      .setInputCols("spell")
      .setOutputCols("prediction")

    new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        spell,
        finisher
      ))
  }

  private def correctMisspells(spellCheckerModel: PipelineModel, testFile: String): Dataset[_] = {
    println("Prediction DataSet")
    val testDataSet = getDataSetFromFile(testFile)
    val predictionDataSet = spellCheckerModel.transform(testDataSet).select("prediction")
    Benchmark.measure("[Norvig Spell Checker] Time to show") {
      predictionDataSet.show()
    }
    predictionDataSet
  }

  private def evaluateSpellChecker(groundTruthFile: String, predictionDataSet: Dataset[_]): Unit = {
    println("Evaluation DataSet")
    val groundTruthDataSet = getGroundTruthDataSet(groundTruthFile)
    val evaluationDataSet = getEvaluationDataSet(predictionDataSet, groundTruthDataSet)
    evaluationDataSet.show(5, false)
    val accuracyDataSet = evaluationDataSet.select(avg(col("accuracy")))
    val accuracy = accuracyDataSet.collect.flatMap(_.toSeq).headOption.getOrElse(-1).toString
    loggingData.logMetric( "accuracy", accuracy.toDouble)
  }

  private def getGroundTruthDataSet(textFile: String): Dataset[_] = {

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

  private def getEvaluationDataSet(predictionDataSet: Dataset[_], groundTruthDataSet: Dataset[_]): Dataset[_] = {
    val evaluationDataSet = predictionDataSet.withColumn("id", monotonically_increasing_id())
      .join(groundTruthDataSet.withColumn("id", monotonically_increasing_id()), Seq("id"))
      .drop("id")
    evaluationDataSet.withColumn("accuracy",
      getAccuracyByRow(col("prediction"), col("ground_truth")))
  }

  private def getAccuracyByRow = udf { (prediction: mutable.WrappedArray[String],
                                        groundTruth: mutable.WrappedArray[String]) =>
    val accuracy = computeAccuracyByRow(prediction.toSet, groundTruth.toSet)
    accuracy
  }

  private def computeAccuracyByRow(prediction: Set[String], groundTruth: Set[String]): Float = {
    val numberOfCorrectWords = prediction.intersect(groundTruth).size.toFloat
    val accuracy: Float = numberOfCorrectWords / groundTruth.size.toFloat
    accuracy
  }

}
