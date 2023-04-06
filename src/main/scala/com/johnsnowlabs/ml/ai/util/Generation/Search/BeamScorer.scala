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
      inputIds: Seq[Array[Long]],
      nextScores: Seq[Array[Float]],
      nextTokens: Seq[Array[Long]],
      nextIndices: Seq[Array[Long]],
      padTokenId: Long,
      eosTokenId: Long,
      beamIndices: Seq[Array[Long]],
      currentLength: Long): (Array[Array[Float]], Array[Array[Long]], Array[Array[Long]])

  def finalize(
      inputIds: Seq[Array[Long]],
      finalBeamScores: Array[Float],
      finalBeamTokens: Array[Long],
      finalBeamIndices: Array[Long],
      maxLength: Long,
      padTokenId: Long,
      eosTokenId: Long,
      beamIndices: Seq[Array[Long]]): (Array[Array[Long]], Array[Float], Array[Array[Long]])
  def getBeamHypothesesSeq: Seq[BeamHypotheses]
  def getNumBeams: Int
  def isDone: Boolean
}
