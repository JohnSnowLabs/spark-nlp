package com.johnsnowlabs.ml.ai.util.Generation

abstract class BeamScorer() {

  protected def process(
      inputIds: Seq[Array[Int]],
      nextScores: Seq[Array[Float]],
      nextTokens: Seq[Array[Int]],
      nextIndices: Seq[Array[Int]]): Seq[Array[Int]]

  protected def finalize(
      inputIds: Seq[Array[Int]],
      nextScores: Seq[Array[Float]],
      nextTokens: Seq[Array[Int]],
      nextIndices: Seq[Array[Int]],
      maxLength: Int): Array[Float]
}
