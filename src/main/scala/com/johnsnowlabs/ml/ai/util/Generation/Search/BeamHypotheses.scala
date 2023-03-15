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

package com.johnsnowlabs.ml.ai.util.Generation.Search

class BeamHypotheses(
    var lengthPenalty: Double,
    var numBeams: Int,
    var earlyStopping: Boolean = false) {
  private var beams: Seq[(Double, Array[Int], Array[Int])] = Seq()
  private var worstScore: Double = 1e9

  def length():Int = {
    beams.length
  }

  def getBeams():Seq[(Double, Array[Int], Array[Int])] ={
    this.beams
  }

  /** Add a new hypotheses to the list
    * @param hypotheses
    *   Hypothesis
    * @param sumLogProbs
    *   Sum of Log Probabilities
    * @param beamIndices
    *   Beam Indices
    */
  def add(hypotheses: Array[Int], sumLogProbs: Double, beamIndices: Array[Int]): Unit = {
    val score = sumLogProbs / Math.pow(hypotheses.length, this.lengthPenalty)
    if (this.beams.length < this.numBeams || score > this.worstScore) {
      this.beams = beams :+ (score, hypotheses, beamIndices)
      if (this.beams.length > this.numBeams) {
        val sortedNextScores = this.beams.zipWithIndex.sortBy(_._1._1)(Ordering.Double.reverse)
        this.worstScore = sortedNextScores.head._1._1
        this.beams = this.beams.zipWithIndex.filter(_._2 != sortedNextScores.head._2).map(_._1)
      } else {
        this.worstScore = Math.min(score, this.worstScore)
      }
    }
  }

  /** If there are enough hypotheses and that none of the hypotheses being generated can become
    * better than the worst one in the heap, then we are done with this sentence.
    *
    * @param bestSumLogProbs
    *   Best Sum of Log Probabilities
    * @param currentLength
    *   Current Length
    * @return
    *   Status of the sentence
    */
  def isDone(bestSumLogProbs: Double, currentLength: Int): Boolean = {
    if (this.beams.length < this.numBeams) {
      false
    } else if (this.earlyStopping) {
      true
    } else {
      val currentScore = bestSumLogProbs / Math.pow(currentLength, this.lengthPenalty)
      this.worstScore >= currentScore
    }
  }
}
