/*
 * Copyright 2017-2021 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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