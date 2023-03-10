package com.johnsnowlabs.ml.ai.util.Generation

//class BeamSearchScorer(
//    var beamSize: Int,
//    var batchSize: Int,
//    var lengthPenalty: Double = 1.0,
//    var doEarlyStopping: Boolean = false,
//    var numBeamHypothesisToKeep: Int = 1)
//    extends BeamScorer {
//
//  private var beamHypothesesSeq: Seq[BeamHypotheses] = Seq.empty[BeamHypotheses]
//  (1 to batchSize) foreach (i =>
//    beamHypothesesSeq = beamHypothesesSeq :+ new BeamHypotheses(
//      lengthPenalty = lengthPenalty,
//      numBeams = beamSize,
//      earlyStopping = doEarlyStopping))
//  private var done: Seq[Boolean] = Seq.iterate(false,batchSize)(_=>false)
//
//  override protected def process(
//      inputIds: Seq[Array[Int]],
//      nextScores: Seq[Array[Float]],
//      nextTokens: Seq[Array[Int]],
//      nextIndices: Seq[Array[Int]],
//      padTokenId: Int,
//      eosTokenId: Int): Seq[Array[Int]] = {
//    val currentLength = inputIds.length
//    val batchSize = this.beamHypothesesSeq.length
//
//  }
//
//  override protected def finalize(
//      inputIds: Seq[Array[Int]],
//      nextScores: Seq[Array[Float]],
//      nextTokens: Seq[Array[Int]],
//      nextIndices: Seq[Array[Int]],
//      maxLength: Int): Array[Float] = ???
//}
