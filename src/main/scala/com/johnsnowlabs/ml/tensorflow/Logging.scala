package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.util.io.OutputHelper
import org.slf4j.LoggerFactory

/* Logging for the TensorFlow Models, probably can be used in other places */
trait Logging {

  def getLogName: String = this.getClass.toString

  protected val logger = LoggerFactory.getLogger(getLogName)
  val verboseLevel: Verbose.Value

  protected def log(value: => String, minLevel: Verbose.Level): Unit = {
    if (minLevel.id >= verboseLevel.id) {
      logger.info(value)
    }
  }
  protected def outputLog(value: => String, uuid: String, shouldLog: Boolean, outputLogsPath: String): Unit = {
    if (shouldLog) {
      OutputHelper.writeAppend(uuid, value, outputLogsPath)
    }
  }
}