package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp.annotators.ner.Verbose
import org.slf4j.LoggerFactory


trait NerDLLogger {

  def verboseLevel: Verbose.Level

  val logger = LoggerFactory.getLogger("NerDL")

  def log(value: => String, minLevel: Verbose.Level): Unit = {
    if (minLevel.id >= verboseLevel.id) {
      logger.info(value)
    }
  }
}
