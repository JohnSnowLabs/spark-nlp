package com.johnsnowlabs.nlp.eval.util

import java.util.NoSuchElementException

import com.johnsnowlabs.nlp.SparkNLP
import com.johnsnowlabs.nlp.annotator.{NerCrfModel, NerDLApproach, NerDLModel}
import com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.annotators.spell.norvig.{NorvigSweetingApproach, NorvigSweetingModel}
import com.johnsnowlabs.nlp.annotators.spell.symmetric.{SymmetricDeleteApproach, SymmetricDeleteModel}
import org.mlflow.api.proto.Service.{RunInfo, RunStatus}
import org.mlflow.tracking.MlflowClient
import org.slf4j.LoggerFactory

class LoggingData(sourceType: String, sourceName: String, experimentName: String) {

  private val logger = LoggerFactory.getLogger("LoggingData")

  private val mlFlowClient = getMLFlowClient
  private val runInfo = getRunInfo(experimentName)
  private val runId: String = getRunId(runInfo)
  private val UNSUPPORTED_SYMBOLS = "[!$%^&*()+|~=`{}\\[\\]:\";'<>?,]"

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

  private def getRunInfo(experimentName: String): Option[RunInfo] = {
    try {
      val expId = getOrCreateExperimentId(mlFlowClient.get, experimentName)
      Some(mlFlowClient.get.createRun(expId))
    } catch {
      case e: Exception =>
        logger.warn("MlflowClient is not running")
        None
    }
  }

  def getOrCreateExperimentId(client: MlflowClient, experimentName: String) : String = {
    val opt = client.getExperimentByName(experimentName)
    opt.isPresent match {
      case true => opt.get().getExperimentId
      case _ => client.createExperiment(experimentName)
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
      mlFlowClient.get.setTag(runId, "mlflow.runName", "Spark NLP " + SparkNLP.currentVersion)
      mlFlowClient.get.setTag(runId, "mlflow.source.type", sourceType)
      mlFlowClient.get.setTag(runId, "mlflow.source.name", sourceName)
    } else {
      println("Spark NLP " + SparkNLP.currentVersion)
    }
  }

  def logParameters(annotator: Any): Unit = {
    val params = getParams(annotator).flatten
    params.foreach{ param =>
      if (param._1 != "inputCols" && param._1 != "labelColumn" && param._1 != "outputCol") {
        logParam(param._1, param._2)
      }
    }
  }

  private def logParam(paramName: String, paramValue: String): Unit = {
    if (runId != "console") {
      mlFlowClient.get.logParam(runId, paramName, paramValue)
    } else {
      println(s"$paramName: $paramValue")
    }
  }

  def getParams(annotator: Any): Array[Option[(String, String)]] = {
    annotator match {
      case nerDLApproach: NerDLApproach =>
        nerDLApproach.params.map{ param =>
          try {
            Some(param.name, nerDLApproach.getOrDefault(param).toString)
          } catch {
            case e: NoSuchElementException => None
          }
        }
      case nerDLModel: NerDLModel =>
        nerDLModel.params.map{ param =>
          try {
            Some(param.name, nerDLModel.getOrDefault(param).toString)
          } catch {
            case e: NoSuchElementException => None
          }
        }
      case nerCrfApproach: NerCrfApproach =>
        nerCrfApproach.params.map{ param =>
          try {
            Some(param.name, nerCrfApproach.getOrDefault(param).toString)
          } catch {
            case e: NoSuchElementException => None
          }
        }
      case nerCrfModel: NerCrfModel =>
        nerCrfModel.params.map{ param =>
          try {
            Some(param.name, nerCrfModel.getOrDefault(param).toString)
          } catch {
            case e: NoSuchElementException => None
          }
        }
      case norvigSweetingApproach: NorvigSweetingApproach =>
        norvigSweetingApproach.params.map{ param =>
          try {
            Some(param.name, norvigSweetingApproach.getOrDefault(param).toString)
          } catch {
            case e: NoSuchElementException => None
          }
        }
      case norvigSweetingModel: NorvigSweetingModel =>
        norvigSweetingModel.params.map{ param =>
          try {
            Some(param.name, norvigSweetingModel.getOrDefault(param).toString)
          } catch {
            case e: NoSuchElementException => None
          }
        }
      case symmetricDeleteApproach: SymmetricDeleteApproach =>
        symmetricDeleteApproach.params.map{ param =>
          try {
            Some(param.name, symmetricDeleteApproach.getOrDefault(param).toString)
          } catch {
            case e: NoSuchElementException => None
          }
        }
      case symmetricDeleteModel: SymmetricDeleteModel =>
        symmetricDeleteModel.params.map{ param =>
          try {
            Some(param.name, symmetricDeleteModel.getOrDefault(param).toString)
          } catch {
            case e: NoSuchElementException => None
          }
        }
      case perceptronApproach: PerceptronApproach =>
        perceptronApproach.params.map{ param =>
          try {
            Some(param.name, perceptronApproach.getOrDefault(param).toString)
          } catch {
            case e: NoSuchElementException => None
          }
        }
      case perceptronModel: PerceptronModel =>
        perceptronModel.params.map{ param =>
          try {
            Some(param.name, perceptronModel.getOrDefault(param).toString)
          } catch {
            case e: NoSuchElementException => None
          }
        }
    }
  }

  def logMetric(metric: String, value: Double): Unit = {
    val roundValue = BigDecimal(value).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
    if (runId != "console") {
      val pattern = UNSUPPORTED_SYMBOLS.r
      val value = pattern.findFirstIn(metric).getOrElse("")
      if (value == "") {
        mlFlowClient.get.logMetric(runId, metric, roundValue)
      } else {
        mlFlowClient.get.logMetric(runId, "SYMBOL", roundValue)
      }
    } else {
      println(metric + ": " + roundValue)
    }
  }

  def closeLog(): Unit = {
    if (runId != "console") {
      mlFlowClient.get.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
    }
  }

}
