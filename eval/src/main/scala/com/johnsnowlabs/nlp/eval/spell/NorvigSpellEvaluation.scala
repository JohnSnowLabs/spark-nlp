package com.johnsnowlabs.nlp.eval.spell

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.eval.util.LoggingData
import com.johnsnowlabs.util.{Benchmark, PipelineModels}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.mutable

class NorvigSpellEvaluation(testFile: String, groundTruthFile: String) {

  private var loggingData = new LoggingData("LOCAL", this.getClass.getSimpleName, "Spell Checkers")

  private case class NorvigSpellEvalConfig(trainFile: String, testFile: String, groundTruthFile: String,
                                           approach: NorvigSweetingApproach, model: NorvigSweetingModel)

  def computeAccuracyAnnotator(trainFile: String, spell: NorvigSweetingApproach): Unit = {
    loggingData.logNorvigParams(spell)
    val norvigSpellEvalConfig = NorvigSpellEvalConfig(trainFile, testFile, groundTruthFile, spell, null)
    computeAccuracy(norvigSpellEvalConfig)
    loggingData.closeLog()
  }

  def computeAccuracyAnnotator(trainFile: String, inputCols: Array[String], outputCol: String, dictionary: String): Unit = {
   val spell = new NorvigSweetingApproach()
     .setInputCols(inputCols)
     .setOutputCol(outputCol)
     .setDictionary(dictionary)
    computeAccuracyAnnotator(trainFile, spell)
  }

  def computeAccuracyModel(spell: NorvigSweetingModel): Unit = {
    loggingData = new LoggingData("LOCAL", this.getClass.getSimpleName, "Spell Checkers")
    loggingData.logNorvigParams(spell)
    val norvigSpellEvalConfig = NorvigSpellEvalConfig("", testFile, groundTruthFile, null, spell)
    computeAccuracy(norvigSpellEvalConfig)
    loggingData.closeLog()
  }

  private def computeAccuracy(norvigSpellEvalConfig: NorvigSpellEvalConfig): Unit = {
    val spellCheckerModel = trainSpellChecker(norvigSpellEvalConfig)
    val predictionDataSet = correctMisspells(spellCheckerModel, testFile)
    evaluateSpellChecker(groundTruthFile, predictionDataSet)
  }

  private def trainSpellChecker(norvigSpellEvalConfig: NorvigSpellEvalConfig): PipelineModel = {
    val trainingDataSet = if (norvigSpellEvalConfig.model == null) getDataSetFromFile(norvigSpellEvalConfig.trainFile)
                          else PipelineModels.dummyDataset
    var spellCheckerModel: PipelineModel = null
    val spellCheckerPipeline = getSpellCheckerPipeline(norvigSpellEvalConfig)
    Benchmark.setPrint(false)
    val time = Benchmark.measure(1, false, "[Norvig Spell Checker] Time to train") {
      spellCheckerModel = spellCheckerPipeline.fit(trainingDataSet)
    }
    if (norvigSpellEvalConfig.model == null) {
      loggingData.logMetric("Training time/s", time)
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

  private def getSpellCheckerPipeline(norvigSpellEvalConfig: NorvigSpellEvalConfig): Pipeline =  {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val finisher = new Finisher()
      .setInputCols("checked")
      .setOutputCols("prediction")

    if (norvigSpellEvalConfig.model == null ) {
      new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          norvigSpellEvalConfig.approach,
          finisher
        ))
    } else {

      val normalizer = new Normalizer()
        .setInputCols("token")
        .setOutputCol("normal")

      new Pipeline()
        .setStages(Array(
          documentAssembler,
          tokenizer,
          normalizer,
          norvigSpellEvalConfig.model,
          finisher
        ))
    }

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
    val numberOfCorrectWords = prediction.intersect(groundTruth).size.toFloat
    numberOfCorrectWords / groundTruth.size.toFloat
  }

}
