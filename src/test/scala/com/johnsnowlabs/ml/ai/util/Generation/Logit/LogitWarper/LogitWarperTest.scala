package com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitWarper

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class LogitWarperTest extends AnyFlatSpec {

  "TopKLogitWarper" should "process correctly" taggedAs FastTest in {
    val vocabSize = 10
    val topK = 5

    val logitWarper = new TopKLogitWarper(k = topK, minTokensToKeep = 1)
    val scoresBatches: Array[Array[Float]] =
      Array(Array(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f))

    val processedScores = logitWarper.call(Seq.empty, scoresBatches, 1).head

    // Check that the top 5 scores are the same and the rest are -inf
    assert(processedScores(0) == Float.NegativeInfinity)
    assert(processedScores(1) == Float.NegativeInfinity)
    assert(processedScores(2) == Float.NegativeInfinity)
    assert(processedScores(3) == Float.NegativeInfinity)
    assert(processedScores(4) == Float.NegativeInfinity)
    assert(processedScores(5) == 0.6f)
    assert(processedScores(6) == 0.7f)
    assert(processedScores(7) == 0.8f)
    assert(processedScores(8) == 0.9f)
    assert(processedScores(9) == 1.0f)

  }

  "TemperatureLogitWarper" should "process correctly" taggedAs FastTest in {
    val vocabSize = 10
    val temperature = 0.5f

    val logitWarper = new TemperatureLogitWarper(temperature = temperature)
    val scoresBatches: Array[Array[Float]] =
      Array(Array(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f))

    val processedScores = logitWarper.call(Seq.empty, scoresBatches, 1).head

    // Check that the scores are correctly scaled
    processedScores.zipWithIndex.foreach({ case (score, i) =>
      assert(score == scoresBatches(0)(i) / temperature)
    })

  }

  "TopPLogitWarper" should "process correctly" taggedAs FastTest in {
    val vocabSize = 10
    val topP = 0.5f

    val logitWarper = new TopPLogitWarper(p = topP, minTokensToKeep = 1)
    val scoresBatches: Array[Array[Float]] =
      Array(Array(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, Float.NegativeInfinity))

    val processedScores = logitWarper.call(Seq.empty, scoresBatches, 1).head

    // print out the processed scores
    processedScores.foreach(println)

    // Check that the top 5 scores are the same and the rest are -inf
    assert(processedScores(0) == Float.NegativeInfinity)
    assert(processedScores(1) == Float.NegativeInfinity)
    assert(processedScores(2) == Float.NegativeInfinity)
    assert(processedScores(3) == Float.NegativeInfinity)
    assert(processedScores(4) == Float.NegativeInfinity)
    assert(processedScores(5) !== Float.NegativeInfinity)
    assert(processedScores(6) !== Float.NegativeInfinity)
    assert(processedScores(7) !== Float.NegativeInfinity)
    assert(processedScores(8) !== Float.NegativeInfinity)
    assert(processedScores(9) == Float.NegativeInfinity)
  }
}
