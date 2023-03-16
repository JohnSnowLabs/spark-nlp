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

  def process(
      inputIds: Seq[Array[Int]],
      nextScores: Seq[Array[Float]],
      nextTokens: Seq[Array[Int]],
      nextIndices: Seq[Array[Int]],
      padTokenId: Int,
      eosTokenId: Int,
      beamIndices: Seq[Array[Int]],
      currentLength: Int): (Array[Array[Float]], Array[Array[Int]], Array[Array[Int]])

  def finalize(
      inputIds: Seq[Array[Int]],
      finalBeamScores: Array[Float],
      finalBeamTokens: Array[Int],
      finalBeamIndices: Array[Int],
      maxLength: Int,
      padTokenId: Int,
      eosTokenId: Int,
      beamIndices: Seq[Array[Int]]):(Array[Array[Int]], Array[Float], Array[Array[Int]])
  def getBeamHypothesesSeq:Seq[BeamHypotheses]
  def getNumBeams:Int
  def isDone:Boolean
}