package com.johnsnowlabs.ml.ai.util.Generation

class BeamHypotheses(
    var lengthPenalty: Double,
    var numBeams: Int,
    var earlyStopping: Boolean = false) {
  private var beams: Seq[(Double, Array[Int], Array[Int])] = Seq()
  private var worstScore: Double = 1e9

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
