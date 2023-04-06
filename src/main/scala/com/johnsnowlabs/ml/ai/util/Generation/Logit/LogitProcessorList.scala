/*
 * Copyright 2017 - 2023  John Snow Labs
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.johnsnowlabs.ml.ai.util.Generation.Logit
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcess.LogitProcessor
import com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitWarper.LogitWarper
class LogitProcessorList {
  private var logitProcesses: List[Logit] = List()

  def addProcess(process: Logit): Unit = {
    logitProcesses = logitProcesses :+ process
  }

  def process(
      inputIds: Seq[Array[Long]],
      scores: Array[Array[Float]],
      currentLength: Int): Array[Array[Float]] = {
    var tempScores = scores
    logitProcesses.foreach(p => {
      if (p.isInstanceOf[LogitProcessor]) {
        tempScores = p.call(inputIds, tempScores, currentLength)
      }
    })
    tempScores
  }

  def warp(
      inputIds: Seq[Array[Long]],
      scores: Array[Array[Float]],
      currentLength: Int): Array[Array[Float]] = {
    var tempScores = scores
    logitProcesses.foreach(p => {
      if (p.isInstanceOf[LogitWarper]) {
        tempScores = p.call(inputIds, tempScores, currentLength)
      }
    })
    tempScores
  }
}
