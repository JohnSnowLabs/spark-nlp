package com.johnsnowlabs.nlp.eval.ner

import com.johnsnowlabs.nlp.eval.util.LoggingData
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object NerMetrics {

   def computeAccuracy(metrics: MulticlassMetrics, loggingData: LoggingData): Unit = {
    val accuracy = (metrics.accuracy * 1000).round / 1000.toDouble
    val weightedPrecision = (metrics.weightedPrecision * 1000).round / 1000.toDouble
    val weightedRecall = (metrics.weightedRecall * 1000).round / 1000.toDouble
    val weightedFMeasure = (metrics.weightedFMeasure * 1000).round / 1000.toDouble
    val weightedFalsePositiveRate = (metrics.weightedFalsePositiveRate * 1000).round / 1000.toDouble
    loggingData.logMetric("Accuracy", accuracy)
    loggingData.logMetric("Weighted Precision", weightedPrecision)
    loggingData.logMetric("Weighted Recall", weightedRecall)
    loggingData.logMetric("Weighted F1 Score", weightedFMeasure)
    loggingData.logMetric("Weighted False Positive Rate", weightedFalsePositiveRate)
  }

  def computeAccuracyByEntity(metrics: MulticlassMetrics, labels: List[String], loggingData: LoggingData): Unit = {
    val predictedLabels = metrics.labels
    predictedLabels.foreach { predictedLabel =>
      val entity = labels(predictedLabel.toInt)
      val precision = (metrics.precision(predictedLabel) * 1000).round / 1000.toDouble
      val recall = (metrics.recall(predictedLabel) * 1000).round / 1000.toDouble
      val f1Score = (metrics.fMeasure(predictedLabel) * 1000).round / 1000.toDouble
      val falsePositiveRate = (metrics.falsePositiveRate(predictedLabel) * 1000).round / 1000.toDouble
      loggingData.logMetric(entity + " Precision", precision)
      loggingData.logMetric(entity + " Recall", recall)
      loggingData.logMetric(entity + " F1-Score", f1Score)
      loggingData.logMetric(entity + " FPR", falsePositiveRate)
    }
  }

  def computeMicroAverage(metrics: MulticlassMetrics, loggingData: LoggingData): Unit = {
    var totalP = 0.0
    var totalR = 0.0
    var totalClassNum = 0

    val labels = metrics.labels

    labels.foreach { label =>
      totalClassNum = totalClassNum + 1
      totalP = totalP + metrics.precision(label)
      totalR = totalR + metrics.recall(label)
    }
    totalP = totalP/totalClassNum
    totalR = totalR/totalClassNum
    val microAverage = 2 * ((totalP*totalR) / (totalP+totalR))
    loggingData.logMetric("Micro-average F1-Score", (microAverage * 1000).round / 1000.toDouble)
  }

}
