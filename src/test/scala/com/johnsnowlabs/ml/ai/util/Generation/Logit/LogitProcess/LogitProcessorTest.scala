package com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcess

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class LogitProcessorTest extends AnyFlatSpec {

  "SuppressLogitProcessor" should "process correctly" taggedAs FastTest in {
    val vocabSize = 3
    val scoresBatches: Array[Array[Float]] = Array(Array.fill(vocabSize)(1.0f))

    // Always suppress
    val suppressedTokenIdx = 0
    val suppressLogitProcessor = new SuppressLogitProcessor(Array(suppressedTokenIdx))

    val processedScores =
      suppressLogitProcessor.call(Seq.empty, scoresBatches, scoresBatches.head.length).head

    assert(processedScores(suppressedTokenIdx) == Float.NegativeInfinity)
  }

  "SuppressLogitProcessor" should "process correctly if beginning token ids specified" taggedAs FastTest in {
    val vocabSize = 3
    val scoresBatches: Array[Array[Float]] = Array(Array.fill(vocabSize)(1.0f))

    // Only Suppress at beginning
    val firstGeneratedIdx =
      Some(1) // Assume we started with only a bos token and no other forced tokens
    val suppressedTokenIdx = 0
    val suppressLogitProcessor =
      new SuppressLogitProcessor(Array(suppressedTokenIdx), firstGeneratedIdx)

    val processedScores =
      suppressLogitProcessor.call(Seq.empty, scoresBatches, firstGeneratedIdx.get).head

    assert(processedScores(suppressedTokenIdx) == Float.NegativeInfinity)

    // Assume we already more tokens
    val processedScoresAfter =
      suppressLogitProcessor.call(Seq.empty, scoresBatches, firstGeneratedIdx.get + 1).head

    assert(processedScoresAfter(suppressedTokenIdx) == 1.0f)
  }

  "ForceTokenLogitProcessor" should "process correctly" taggedAs FastTest in {
    val taskTokenId = 1 // Example token id for setting the task

    val forcedIdx = 1
    val forcedTokenIds = Array((forcedIdx, taskTokenId))

    val vocabSize = 2
    val scoresBatches: Array[Array[Float]] = Array(Array.fill(vocabSize)(0.5f))

    val forceTokenLogitProcessor = new ForcedTokenLogitProcessor(forcedTokenIds)

    val forcedScores = forceTokenLogitProcessor.call(Seq.empty, scoresBatches, 1).head

    assert(forcedScores(taskTokenId) == Float.PositiveInfinity)
    assert(forcedScores(0) == 0)

    // Multiple forced tokens for same index, should force token idx 0
    val forceTokenLogitProcessorMultiple =
      new ForcedTokenLogitProcessor(forcedTokenIds :+ (forcedIdx, 0))

    val forcedScoresMultiple =
      forceTokenLogitProcessorMultiple.call(Seq.empty, scoresBatches, 1).head

    assert(forcedScoresMultiple(0) == Float.PositiveInfinity)
    assert(forcedScoresMultiple(1) == 0)
  }

}
