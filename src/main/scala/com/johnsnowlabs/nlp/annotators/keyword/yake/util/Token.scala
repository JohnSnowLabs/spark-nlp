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

package com.johnsnowlabs.nlp.annotators.keyword.yake.util

import scala.collection.mutable
import scala.math.{log, max}

class Token(var token: String,
            var termFrequency: Int,
            var totalSentences: Int,
            var meanTF: Double,
            var stdTF: Double,
            var maxTF: Double,
            var leftCO: mutable.Map[String, Int],
            var rightCO: mutable.Map[String, Int]) {
  var nCount = 0
  var aCount = 0
  var medianSentenceOffset = 0
  var numberOfSentences = 0

  def TCase(): Double = {
    max(nCount, aCount).toDouble / (1 + log(termFrequency))
  }

  def TPosition(): Double = {
    log(3 + medianSentenceOffset)
  }

  def TFNorm(): Double = {
    termFrequency.toDouble / (meanTF + stdTF)
  }

  def TSentence(): Double = {
    numberOfSentences.toDouble / totalSentences.toDouble
  }

  def TRel(): Double = {
    1.0 + ((if (leftCO.isEmpty) 0.0 else (leftCO.size.toDouble / leftCO.values.sum.toDouble))
      + (if (rightCO.isEmpty) 0.0 else (leftCO.size.toDouble / rightCO.values.sum.toDouble)))* (termFrequency.toDouble / maxTF.toDouble)
  }

  def TScore(): Double = {
    TPosition() * TRel() / (TCase() + (TFNorm() / TRel()) + (TSentence() / TRel()))
  }
}
