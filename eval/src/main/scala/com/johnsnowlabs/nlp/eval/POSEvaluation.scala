package com.johnsnowlabs.nlp.eval

import com.johnsnowlabs.nlp.annotator.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.eval.util.{GoldTokenizer, LoggingData, TagsMetrics}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.mutable

class POSEvaluation(sparkSession: SparkSession, testFile: String) {

  import sparkSession.implicits._

  val goldTokenizer = new GoldTokenizer(sparkSession)
  val loggingData = new LoggingData("LOCAL", this.getClass.getSimpleName, "Part of Speech Tagger")

  private case class PosEvalConfiguration(trainFile: String, posModel: PerceptronModel, posApproach: PerceptronApproach)

  def computeAccuracyModel(posModel: PerceptronModel): Unit = {
    val posEvalConfiguration = PosEvalConfiguration("", posModel, null)
    computeAccuracy(posEvalConfiguration)
    loggingData.closeLog()
  }

  def computeAccuracy(posEvalConfiguration: PosEvalConfiguration): Unit = {
    val posLabels = getPosLabels(posEvalConfiguration)
    val evaluationDataSet = getEvaluationDataSet(posEvalConfiguration, posLabels)
    evaluationDataSet.show(5, false)
    val predictionLabelsRDD = evaluationDataSet.select("predictionIndex", "labelIndex")
      .map(r => (r.getDouble(0), r.getDouble(1)))
    val metrics = new MulticlassMetrics(predictionLabelsRDD.rdd)
    TagsMetrics.computeAccuracy(metrics, loggingData)
    TagsMetrics.computeAccuracyByEntity(metrics, posLabels, loggingData)
    TagsMetrics.computeMicroAverage(metrics,loggingData)
  }

  private def getPosLabels(posEvalConfiguration: PosEvalConfiguration): List[String] = {
    val joinedDataSet = getJoinedDataSet(posEvalConfiguration)
    val labels: Seq[String] = joinedDataSet.select($"testPOS").collect.flatMap(_.toSeq).flatMap {
      entitiesArray =>
        entitiesArray.asInstanceOf[mutable.WrappedArray[String]]
    }
    labels.toList.distinct
  }

  private def getJoinedDataSet(posEvalConfiguration: PosEvalConfiguration): Dataset[_] = {
    val testDataSet = goldTokenizer.getTestTokensTagsDataSet(testFile)
    val predictionDataSet = getPredictionDataSet(posEvalConfiguration)

    predictionDataSet
      .join(testDataSet, Seq("id"))
      .withColumn("predictedTokensLength", goldTokenizer.calLengthOfArray($"predictedTokens"))
      .withColumn("predictedPOSLength", goldTokenizer.calLengthOfArray($"predictedPOS"))
      .withColumn("testTokensLength", goldTokenizer.calLengthOfArray($"testTokens"))
      .withColumn("testPOSLength", goldTokenizer.calLengthOfArray($"testPOS"))
      .withColumn("tokensDiffFromTest", $"testTokensLength" - $"predictedTokensLength")
      .withColumn("posDiffFromTest", $"testPOSLength" - $"predictedPOSLength")
      .withColumn("missingTokens", goldTokenizer.extractMissingTokens($"testTokens", $"predictedTokens"))
      .withColumn("missingPOS", goldTokenizer.extractMissingTokens($"testPOS", $"predictedPOS"))
      .withColumn("equalPOS", col("predictedPOSLength") === col("testPOSLength"))

  }

  private def getPredictionDataSet(posEvalConfiguration: PosEvalConfiguration): Dataset[_] = {
    val testDataSet = goldTokenizer.getGoldenTokenizer(testFile)
    //val trainDataSet = POS().readDataset(sparkSession, posEvalConfiguration.trainFile)
    val posModel = posEvalConfiguration.posModel
      .setInputCols("document", "token")
      .setOutputCol("pos")

    val predictionDataSet = posModel.transform(testDataSet)

    predictionDataSet.select(
      $"id",
      $"document",
      $"token",
      $"token.result".alias("predictedTokens"),
      $"pos.result".alias("predictedPOS")
    )
  }

  private def getEvaluationDataSet(posEvalConfiguration: PosEvalConfiguration, labels: List[String]): Dataset[_] = {
    val joinedDataSet = getJoinedDataSet(posEvalConfiguration)
    val labelAndPrediction: Seq[(String, String)] = joinedDataSet.select("testPOS", "predictedPOS").rdd.map { row =>
      val labelColumn: Seq[String] = row.get(0).asInstanceOf[mutable.WrappedArray[String]]
      val predictionColumn: Seq[String] = row.get(1).asInstanceOf[mutable.WrappedArray[String]]
      (labelColumn.toList, predictionColumn.toList)
    }.collect().flatMap(row => row._1 zip row._2)

    labelAndPrediction.toDF("label", "prediction")
      .withColumn("labelIndex", getLabelIndex(labels)(col("label")))
      .withColumn("predictionIndex", getLabelIndex(labels)(col("prediction")))
  }

  private def getLabelIndex(labels: List[String]) = udf { label: String =>
    val index = labels.indexOf(label)
    index.toDouble
  }

}
