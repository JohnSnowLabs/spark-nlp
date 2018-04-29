package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.ner.Verbose
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
}