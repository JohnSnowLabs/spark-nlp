package com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcess

/** Forces specific token ids to be sampled at specific indexes. Note that the order of the of the
  * ids is relevant, as only the last forced token at the index will be the result.
  *
  * @param forcedIds
  *   Indexes with their forced token id
  */
class ForcedTokenLogitProcessor(forcedIds: Array[(Int, Int)]) extends LogitProcessor {

  override def call(
      inputIds: Seq[Array[Int]],
      scores: Array[Array[Float]],
      currentLength: Int): Array[Array[Float]] = scores.map { batchScores =>
    var batchArray: Array[Float] = batchScores

    forcedIds.foreach { case (idx, tokenId) =>
      if (idx == currentLength) {
        batchArray = Array.fill(batchScores.length)(0.0f).updated(tokenId, Float.PositiveInfinity)
      }
    }

    batchArray

  }
}
