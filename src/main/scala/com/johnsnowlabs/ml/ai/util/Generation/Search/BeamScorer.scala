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

abstract class BeamScorer() {

  protected def process(
      inputIds: Seq[Array[Int]],
      nextScores: Seq[Array[Double]],
      nextTokens: Seq[Array[Int]],
      nextIndices: Seq[Array[Int]],
      padTokenId: Int,
      eosTokenId: Int,
      beamIndices: Seq[Array[Int]]): (Array[Array[Double]], Array[Array[Int]], Array[Array[Int]])

  protected def finalize(
      inputIds: Seq[Array[Int]],
      finalBeamScores: Array[Double],
      finalBeamTokens: Array[Int],
      finalBeamIndices: Array[Int],
      maxLength: Int,
      padTokenId: Int,
      eosTokenId: Int,
      beamIndices: Seq[Array[Int]]):(Array[Array[Int]], Array[Double], Array[Array[Int]])
  def getBeamHypothesesSeq:Seq[BeamHypotheses]
  def getNumBeams:Int
}