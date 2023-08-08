package com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcess

import scala.collection.mutable

/** Sets the probability to -inf for provided tokenIds, so they are not sampled.
  *
  * @param suppressTokenIds
  *   List of token ids to suppress
  * @param atBeginIdx
  *   Whether to only suppress tokens at the beginning of the generation. The beginning of the
  *   generation is marked the index (e.g. after the bos token and potential forced tokens).
  */
class SuppressLogitProcessor(suppressTokenIds: Array[Int], atBeginIdx: Option[Int] = None)
    extends LogitProcessor {

  private def suppressScores(batchScores: Array[Float]): Array[Float] = {
    val batchArray = mutable.ArrayBuffer(batchScores: _*)

    suppressTokenIds.foreach { tokenId => batchArray.update(tokenId, Float.NegativeInfinity) }
    batchArray.toArray
  }

  override def call(
      inputIds: Seq[Array[Int]],
      scores: Array[Array[Float]],
      currentLength: Int): Array[Array[Float]] = scores.map { batchScores =>
    atBeginIdx match {
      case Some(beginIdx) =>
        if (currentLength == beginIdx) suppressScores(batchScores) else batchScores
      case None => suppressScores(batchScores)
    }
  }

}
