package com.johnsnowlabs.nlp.eval.util

import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
import com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach
import org.mlflow.api.proto.Service.{RunInfo, RunStatus}
import org.mlflow.tracking.MlflowClient
import org.slf4j.LoggerFactory

class LoggingData(sourceType: String, sourceName: String) {

  private val logger = LoggerFactory.getLogger("LoggingData")

  private val mlFlowClient = getMLFlowClient
  private val runInfo = getRunInfo
  private val runId: String = getRunId(runInfo)

  setMLflowTags()

  private def getMLFlowClient: Option[MlflowClient] = {
    val trackingUri: Option[String] = sys.env.get("MLFLOW_TRACKING_URI")
    if (trackingUri.isDefined) {
      Some(new MlflowClient())
    } else {
      logger.warn("MlflowClient requires MLFLOW_TRACKING_URI is set")
      None
    }
  }

  private def getRunInfo: Option[RunInfo] = {
    try {
      Some(mlFlowClient.get.createRun())
    } catch {
      case e: Exception =>
        logger.warn("MlflowClient is not running")
        None
    }
  }

  private def getRunId(runInfo: Option[RunInfo]): String = {
    if (runInfo.isDefined) {
      runInfo.get.getRunUuid
    } else {
      "console"
    }
  }

  private def setMLflowTags(): Unit = {
    if (runId != "console") {
      mlFlowClient.get.setTag(runId, "mlflow.runName", "SparkNLP 2.0.8")
      mlFlowClient.get.setTag(runId, "mlflow.source.type", sourceType)
      mlFlowClient.get.setTag(runId, "mlflow.source.name", sourceName)
    }
  }

  def logNorvigParams(spell: NorvigSweetingApproach): Unit = {
    if (runId != "console") {
      mlFlowClient.get.logParam(runId, "caseSensitive", spell.getCaseSensitive.toString)
      mlFlowClient.get.logParam(runId, "doubleVariants", spell.getDoubleVariants.toString)
      mlFlowClient.get.logParam(runId, "shortCircuit", spell.getShortCircuit.toString)
      mlFlowClient.get.logParam(runId, "frequencyPriority", spell.getFrequencyPriority.toString)
      mlFlowClient.get.logParam(runId, "wordSizeIgnore", spell.getWordSizeIgnore.toString)
      mlFlowClient.get.logParam(runId, "dupsLimit", spell.getDupsLimit.toString)
      mlFlowClient.get.logParam(runId, "reductLimit", spell.getReductLimit.toString)
      mlFlowClient.get.logParam(runId, "intersections", spell.getIntersections.toString)
      mlFlowClient.get.logParam(runId, "vowelSwapLimit", spell.getVowelSwapLimit.toString)
    } else {
      println(s"Parameters for $sourceName:")
      println("caseSensitive: " + spell.getCaseSensitive.toString)
      println("doubleVariants: " + spell.getDoubleVariants.toString)
      println("shortCircuit: " + spell.getShortCircuit.toString)
      println("frequencyPriority: " + spell.getFrequencyPriority.toString)
      println("wordSizeIgnore: " + spell.getWordSizeIgnore.toString)
      println("dupsLimit: " + spell.getDupsLimit.toString)
      println("reductLimit: " + spell.getReductLimit.toString)
      println("intersections: " + spell.getIntersections.toString)
      println("vowelSwapLimit: " + spell.getVowelSwapLimit.toString)
    }
  }

  def logSymSpellParams(spell: SymmetricDeleteApproach): Unit = {
    if (runId != "console") {
      mlFlowClient.get.logParam(runId, "maxEditDistance", spell.getMaxEditDistance.toString)
      mlFlowClient.get.logParam(runId, "frequencyThreshold", spell.getFrequencyThreshold.toString)
      mlFlowClient.get.logParam(runId, "deletesThreshold", spell.getDeletesThreshold.toString)
      mlFlowClient.get.logParam(runId, "dupsLimit", spell.getDupsLimit.toString)
    } else {
      println(s"Parameters for $sourceName:")
      println("maxEditDistance: " + spell.getMaxEditDistance.toString)
      println("frequencyThreshold: " + spell.getFrequencyThreshold.toString)
      println("deletesThreshold: " + spell.getDeletesThreshold.toString)
      println("dupsLimit: " + spell.getDupsLimit.toString)
    }
  }

  def logMetric(metric: String, value: Double): Unit = {
    val roundValue = BigDecimal(value).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
    if (runId != "console") {
      mlFlowClient.get.logMetric(runId, metric, roundValue)
    } else {
      println(s"Accuracy Metrics for $sourceName:")
      println(metric + ": " + roundValue)
    }
  }

  def closeLog(): Unit = {
    if (runId != "console") {
      mlFlowClient.get.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
    }
  }

}
