package com.johnsnowlabs.ml.ai.util.Generation

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
}
