/*
 * Copyright 2017-2026 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.johnsnowlabs.nlp.annotators.uncertainty

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class UncertaintyMetricsTest extends AnyFlatSpec {

  private val tolerance = 1e-4

  private def approxEqual(a: Double, b: Double, tol: Double = tolerance): Boolean =
    math.abs(a - b) < tol

  behavior of "UncertaintyMetrics.cosineSimilarity"

  it should "return 1.0 for identical vectors" taggedAs FastTest in {
    val v = Array(1.0f, 2.0f, 3.0f)
    assert(approxEqual(UncertaintyMetrics.cosineSimilarity(v, v), 1.0))
  }

  it should "return 0.0 for orthogonal vectors" taggedAs FastTest in {
    val a = Array(1.0f, 0.0f)
    val b = Array(0.0f, 1.0f)
    assert(approxEqual(UncertaintyMetrics.cosineSimilarity(a, b), 0.0))
  }

  it should "return -1.0 for opposite vectors" taggedAs FastTest in {
    val a = Array(1.0f, 0.0f)
    val b = Array(-1.0f, 0.0f)
    assert(approxEqual(UncertaintyMetrics.cosineSimilarity(a, b), -1.0))
  }

  it should "return 0.0 for a zero vector rather than throwing" taggedAs FastTest in {
    val a = Array(0.0f, 0.0f)
    val b = Array(1.0f, 1.0f)
    assert(approxEqual(UncertaintyMetrics.cosineSimilarity(a, b), 0.0))
  }

  it should "match a hand-computed value for a non-trivial pair" taggedAs FastTest in {
    val a = Array(1.0f, 2.0f)
    val b = Array(2.0f, 3.0f)
    // dot=8, |a|=sqrt(5), |b|=sqrt(13) -> 8/sqrt(65) = 0.99227787671
    assert(approxEqual(UncertaintyMetrics.cosineSimilarity(a, b), 0.99227787671))
  }

  behavior of "UncertaintyMetrics.similarityMatrix"

  it should "have 1.0 on the diagonal and be symmetric" taggedAs FastTest in {
    val vectors = Array(Array(1.0f, 0.0f), Array(0.0f, 1.0f), Array(1.0f, 1.0f))
    val m = UncertaintyMetrics.similarityMatrix(vectors)
    for (i <- vectors.indices) assert(approxEqual(m(i)(i), 1.0))
    for (i <- vectors.indices; j <- vectors.indices) assert(approxEqual(m(i)(j), m(j)(i)))
  }

  behavior of "UncertaintyMetrics.clusterBySimilarity / groupByCluster"

  it should "put all samples in one cluster when all pairwise similarities are high" taggedAs FastTest in {
    val sim =
      Array(Array(1.0f, 0.99f, 0.98f), Array(0.99f, 1.0f, 0.97f), Array(0.98f, 0.97f, 1.0f))
    val clusterIds = UncertaintyMetrics.clusterBySimilarity(sim, 0.85f)
    assert(clusterIds.distinct.length == 1)
    assert(UncertaintyMetrics.groupByCluster(clusterIds) == Seq(Seq(0, 1, 2)))
  }

  it should "put each sample in its own cluster when all pairwise similarities are low" taggedAs FastTest in {
    val sim = Array(Array(1.0f, 0.1f, 0.2f), Array(0.1f, 1.0f, 0.1f), Array(0.2f, 0.1f, 1.0f))
    val clusterIds = UncertaintyMetrics.clusterBySimilarity(sim, 0.85f)
    assert(clusterIds.distinct.length == 3)
    assert(UncertaintyMetrics.groupByCluster(clusterIds) == Seq(Seq(0), Seq(1), Seq(2)))
  }

  it should "chain transitively across the threshold (single-link)" taggedAs FastTest in {
    // 0-1 similar, 1-2 similar, but 0-2 not directly similar: still one cluster via transitivity.
    val sim = Array(Array(1.0f, 0.9f, 0.1f), Array(0.9f, 1.0f, 0.9f), Array(0.1f, 0.9f, 1.0f))
    val clusterIds = UncertaintyMetrics.clusterBySimilarity(sim, 0.85f)
    assert(clusterIds.distinct.length == 1)
  }

  it should "respect the exact threshold boundary (>=)" taggedAs FastTest in {
    val sim = Array(Array(1.0f, 0.85f), Array(0.85f, 1.0f))
    assert(UncertaintyMetrics.clusterBySimilarity(sim, 0.85f).distinct.length == 1)
    assert(UncertaintyMetrics.clusterBySimilarity(sim, 0.86f).distinct.length == 2)
  }

  behavior of "UncertaintyMetrics.semanticEntropy"

  it should "be exactly 0.0 for a single cluster (all samples equivalent)" taggedAs FastTest in {
    val clusterIds = Array(0, 0, 0, 0)
    assert(UncertaintyMetrics.semanticEntropy(clusterIds) == 0.0)
  }

  it should "equal ln(n) for n samples each in their own cluster with uniform scores" taggedAs FastTest in {
    val n = 4
    val clusterIds = Array(0, 1, 2, 3)
    val se = UncertaintyMetrics.semanticEntropy(clusterIds)
    assert(approxEqual(se, math.log(n.toDouble)))
  }

  it should "match a hand-computed value for a mixed cluster distribution" taggedAs FastTest in {
    // 4 samples, uniform score log(1/4) each; clusters {0,1}, {2}, {3}.
    val clusterIds = Array(0, 0, 2, 3)
    val n = 4
    val uniformScore = -math.log(n.toDouble)
    // cluster {0,1}: logSumExp(uniformScore, uniformScore) = uniformScore + log(2)
    val clusterA = -(uniformScore + math.log(2.0))
    // clusters {2}, {3}: logSumExp of a single value is itself
    val clusterB = -uniformScore
    val clusterC = -uniformScore
    val expected = (clusterA + clusterB + clusterC) / 3.0
    assert(approxEqual(UncertaintyMetrics.semanticEntropy(clusterIds), expected))
  }

  it should "use supplied sequence log-probabilities when given" taggedAs FastTest in {
    val clusterIds = Array(0, 1)
    val logProbs = Array(-0.1, -5.0)
    // Two singleton clusters: SE = (-(-0.1) + -(-5.0)) / 2 = (0.1 + 5.0) / 2
    val expected = (0.1 + 5.0) / 2.0
    assert(approxEqual(UncertaintyMetrics.semanticEntropy(clusterIds, Some(logProbs)), expected))
  }

  behavior of "UncertaintyMetrics.eccentricity"

  it should "be exactly 0.0 for a single sample" taggedAs FastTest in {
    val (aggregate, perSample) = UncertaintyMetrics.eccentricity(Array(Array(1.0)))
    assert(aggregate == 0.0)
    assert(perSample sameElements Array(0.0))
  }

  it should "be exactly 0.0 (up to floating point) when all samples are identical" taggedAs FastTest in {
    val n = 4
    val affinity = Array.fill(n, n)(1.0)
    val (aggregate, perSample) = UncertaintyMetrics.eccentricity(affinity)
    assert(approxEqual(aggregate, 0.0, 1e-6))
    perSample.foreach(s => assert(approxEqual(s, 0.0, 1e-6)))
  }

  it should "be higher for a clear outlier than for samples in the consensus" taggedAs FastTest in {
    // Samples 0,1,2 mutually near-identical; sample 3 is an outlier.
    val affinity = Array(
      Array(1.0, 0.95, 0.95, 0.05),
      Array(0.95, 1.0, 0.95, 0.05),
      Array(0.95, 0.95, 1.0, 0.05),
      Array(0.05, 0.05, 0.05, 1.0))
    val (_, perSample) = UncertaintyMetrics.eccentricity(affinity)
    assert(perSample(3) > perSample(0))
    assert(perSample(3) > perSample(1))
    assert(perSample(3) > perSample(2))
  }

  behavior of "UncertaintyMetrics.alignByCharSpan"

  it should "assign 0.0 importance to a token that overlaps no phrase" taggedAs FastTest in {
    val tokens = Seq((0, 3, 1.0))
    val phrases = Seq((5, 8, 0.9))
    val aligned = UncertaintyMetrics.alignByCharSpan(tokens, phrases)
    assert(aligned == Seq((1.0, 0.0)))
  }

  it should "assign a phrase's importance to a token spanning exactly the same range" taggedAs FastTest in {
    val tokens = Seq((0, 5, 2.0))
    val phrases = Seq((0, 5, 0.75))
    val aligned = UncertaintyMetrics.alignByCharSpan(tokens, phrases)
    assert(aligned == Seq((2.0, 0.75)))
  }

  it should "overlap-weight average when a token spans two phrases" taggedAs FastTest in {
    // token [0,10) overlaps phrase [0,4) (importance 1.0, overlap 4) and phrase [4,10) (importance
    // 0.0, overlap 6) -> weighted average = (4*1.0 + 6*0.0) / 10 = 0.4
    val tokens = Seq((0, 10, 3.0))
    val phrases = Seq((0, 4, 1.0), (4, 10, 0.0))
    val aligned = UncertaintyMetrics.alignByCharSpan(tokens, phrases)
    assert(aligned.length == 1)
    assert(approxEqual(aligned.head._1, 3.0))
    assert(approxEqual(aligned.head._2, 0.4))
  }

  behavior of "UncertaintyMetrics.marsScore"

  it should "be exactly 0.0 for an empty sequence" taggedAs FastTest in {
    assert(UncertaintyMetrics.marsScore(Seq.empty) == 0.0)
  }

  it should "equal the plain mean NLL when all tokens have equal importance" taggedAs FastTest in {
    val nllAndImportance = Seq((1.0, 0.5), (2.0, 0.5), (3.0, 0.5))
    // Equal importance -> softmax is uniform -> importance-weighted mean == plain mean == 2.0
    val expected = 2.0
    assert(approxEqual(UncertaintyMetrics.marsScore(nllAndImportance), expected))
  }

  it should "weight the highest-importance token's NLL more heavily" taggedAs FastTest in {
    // One token with much higher importance than the rest should pull the score towards its NLL.
    val highImportanceHighNll =
      UncertaintyMetrics.marsScore(Seq((1.0, 0.0), (1.0, 0.0), (10.0, 100.0)))
    val allEqualImportance =
      UncertaintyMetrics.marsScore(Seq((1.0, 1.0), (1.0, 1.0), (10.0, 1.0)))
    assert(highImportanceHighNll > allEqualImportance)
  }

  behavior of "UncertaintyMetrics JSON parsing"

  it should "parse meanLogProb from a well-formed completion_probabilities array" taggedAs FastTest in {
    val json =
      """[{"logprob": -0.5, "top_logprobs": []}, {"logprob": -1.5, "top_logprobs": []}]"""
    assert(UncertaintyMetrics.meanLogProb(json).exists(approxEqual(_, -1.0)))
  }

  it should "not be fooled by logprob fields nested inside top_logprobs" taggedAs FastTest in {
    // Only the top-level logprob (-0.5) should be averaged, not the alternatives' logprobs.
    val json =
      """[{"logprob": -0.5, "top_logprobs": [{"logprob": -0.5}, {"logprob": -9.0}, {"logprob": -20.0}]}]"""
    assert(UncertaintyMetrics.meanLogProb(json).exists(approxEqual(_, -0.5)))
  }

  it should "return None for empty or malformed completion_probabilities JSON" taggedAs FastTest in {
    assert(UncertaintyMetrics.meanLogProb("[]").isEmpty)
    assert(UncertaintyMetrics.meanLogProb("not json").isEmpty)
    assert(UncertaintyMetrics.meanLogProb("""{"not":"an array"}""").isEmpty)
  }

  it should "compute predictiveEntropy over top_logprobs and skip tokens without enough alternatives" taggedAs FastTest in {
    // Token 1: two alternatives with logprob 0 (i.e. prob 1.0 each) -> degenerate but defined.
    // Token 2: only one alternative -> skipped (needs >= 2).
    val json =
      """[{"logprob": -0.1, "top_logprobs": [{"logprob": -0.6931471805599453}, {"logprob": -0.6931471805599453}]},
        | {"logprob": -0.1, "top_logprobs": [{"logprob": -0.1}]}]""".stripMargin
    val entropy = UncertaintyMetrics.predictiveEntropy(json)
    assert(entropy.isDefined)
    // Two alternatives each with p=0.5 -> H = -(0.5*ln(0.5) + 0.5*ln(0.5)) = ln(2)
    assert(approxEqual(entropy.get, math.log(2.0)))
  }

  it should "parse a well-formed token_importance array" taggedAs FastTest in {
    val json =
      """[{"begin": 0, "end": 4, "importance": 0.9}, {"begin": 6, "end": 7, "importance": 0.1}]"""
    val phrases = UncertaintyMetrics.parseTokenImportance(json)
    assert(phrases == Seq((0, 4, 0.9), (6, 7, 0.1)))
  }

  it should "parse token spans and logprobs from completion_probabilities using cumulative UTF-8 byte offsets" taggedAs FastTest in {
    // "Hi!" as three single-byte-per-char tokens.
    val json =
      """[{"logprob": -0.1, "bytes": [72]},
        | {"logprob": -0.2, "bytes": [105]},
        | {"logprob": -0.3, "bytes": [33]}]""".stripMargin
    val spans = UncertaintyMetrics.tokenSpansFromCompletionProbabilities(json)
    assert(
      spans == Seq(
        UncertaintyMetrics.TokenSpan(0, 1, -0.1),
        UncertaintyMetrics.TokenSpan(1, 2, -0.2),
        UncertaintyMetrics.TokenSpan(2, 3, -0.3)))
  }

  it should "correctly offset a multi-byte UTF-8 character split across the bytes array" taggedAs FastTest in {
    // "é" (U+00E9) is 2 bytes in UTF-8 (0xC3 0xA9); one token carries both bytes.
    val json = """[{"logprob": -0.4, "bytes": [195, 169]}, {"logprob": -0.5, "bytes": [33]}]"""
    val spans = UncertaintyMetrics.tokenSpansFromCompletionProbabilities(json)
    assert(spans.length == 2)
    assert(spans.head == UncertaintyMetrics.TokenSpan(0, 1, -0.4)) // "é" is 1 UTF-16 char
    assert(spans(1) == UncertaintyMetrics.TokenSpan(1, 2, -0.5))
  }

  it should "parse a well-formed square entailment_matrix" taggedAs FastTest in {
    val json = "[[1.0, 0.2], [0.3, 1.0]]"
    val matrix = UncertaintyMetrics.parseEntailmentMatrix(json)
    assert(matrix.isDefined)
    assert(matrix.get.length == 2)
    assert(approxEqual(matrix.get(0)(1), 0.2))
    assert(approxEqual(matrix.get(1)(0), 0.3))
  }

  it should "return None for a non-square entailment_matrix" taggedAs FastTest in {
    val json = "[[1.0, 0.2, 0.3], [0.3, 1.0]]"
    assert(UncertaintyMetrics.parseEntailmentMatrix(json).isEmpty)
  }
}
