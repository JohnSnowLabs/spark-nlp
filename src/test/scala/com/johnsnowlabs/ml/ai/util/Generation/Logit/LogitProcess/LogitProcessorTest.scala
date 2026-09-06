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

  "MinlengthLogitProcessor" should "process correctly" taggedAs FastTest in {

    val vocabSize = 32
    val scoresBatches: Array[Array[Float]] = Array(Array.fill(vocabSize)(1.0f))

    val minLength = 2
    val minLengthLogitProcessor = new MinLengthLogitProcessor(
      eosTokenId = vocabSize - 1,
      minLength = minLength,
      vocabSize = vocabSize)

    // if the min length is not reached, the eos token should be suppressed
    val processedScores =
      minLengthLogitProcessor.call(Seq.empty, scoresBatches, minLength - 1).head

    assert(processedScores(vocabSize - 1) == Float.NegativeInfinity)

    // if the min length is reached, the eos token should not be suppressed
    val processedScoresAfter =
      minLengthLogitProcessor.call(Seq.empty, scoresBatches, minLength).head

    assert(processedScoresAfter(vocabSize - 1) == 1.0f)
  }

  "NoRepeatNgramsLogitProcessor" should "ban the token that previously followed a repeated bigram prefix" taggedAs FastTest in {
    val inputIds = Seq(Array(1, 2, 3, 1, 2))
    val vocabSize = 5
    val scores = Array(Array.fill(vocabSize)(1.0f))
    val processor = new NoRepeatNgramsLogitProcessor(noRepeatNgramSize = 2, vocabSize = vocabSize)

    val result = processor.call(inputIds, scores, currentLength = inputIds.head.length)
    println(s"[NoRepeatNgrams debug] result=${result.head.toSeq}")
    assert(
      result.head(3) == Float.NegativeInfinity,
      s"token 3 (which followed the bigram (1,2) before) should be banned, got scores=${result.head.toSeq}")
  }

  "RepetitionPenaltyLogitProcessor" should "divide a positive previously-seen token's logit by the penalty and multiply a negative one" taggedAs FastTest in {
    val inputIds = Seq(Array(0, 1))
    val scores = Array(Array(2.0f, -1.0f, 3.0f))
    val processor = new RepetitionPenaltyLogitProcessor(penalty = 2.0)

    val result = processor.call(inputIds, scores, currentLength = 2)
    println(s"[RepetitionPenalty debug] result=${result.head.toSeq}")
    assert(result.head(0) == 1.0f, s"expected 2.0/2.0=1.0, got ${result.head(0)}")
    assert(result.head(1) == -2.0f, s"expected -1.0*2.0=-2.0, got ${result.head(1)}")
    assert(
      result.head(2) == 3.0f,
      s"token 2 never appeared - must be untouched, got ${result.head(2)}")
  }

  it should "leave scores untouched when the penalty is exactly 1.0 (no-op)" taggedAs FastTest in {
    val inputIds = Seq(Array(0, 1))
    val scores = Array(Array(2.0f, -1.0f, 3.0f))
    val processor = new RepetitionPenaltyLogitProcessor(penalty = 1.0)

    val result = processor.call(inputIds, scores, currentLength = 2)
    assert(result.head.toSeq == Seq(2.0f, -1.0f, 3.0f))
  }
}
