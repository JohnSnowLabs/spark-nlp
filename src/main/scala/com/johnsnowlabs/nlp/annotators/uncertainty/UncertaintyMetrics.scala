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

import breeze.linalg.{DenseMatrix, DenseVector, diag, eigSym}
import com.johnsnowlabs.nlp.annotators.seq2seq.CompletionProbabilities
import org.json4s._
import org.json4s.jackson.JsonMethods._

/** Pure numerical building blocks for [[LLMUncertaintyEstimator]]. Contains no Spark types so
  * that every formula here can be unit tested directly against hand-computed values (and against
  * the reference Python implementations of the papers these formulas come from) without spinning
  * up a SparkSession.
  *
  * Score direction convention used throughout this file: every returned "uncertainty" quantity is
  * oriented so that a higher value means the model was less certain. `0.0` (or the formula's
  * natural minimum) always means maximal certainty (e.g. all samples identical).
  */
object UncertaintyMetrics {

  // ---------------------------------------------------------------------------------------------
  // Similarity / clustering (used by Semantic Entropy)
  // ---------------------------------------------------------------------------------------------

  /** Cosine similarity between two equal-length vectors. Returns `0.0` if either vector is all
    * zeros (rather than dividing by zero).
    */
  def cosineSimilarity(a: Array[Float], b: Array[Float]): Float = {
    require(
      a.length == b.length,
      s"Vectors must have the same length, got ${a.length} and ${b.length}")
    var dot = 0.0
    var normA = 0.0
    var normB = 0.0
    var i = 0
    while (i < a.length) {
      dot += a(i).toDouble * b(i).toDouble
      normA += a(i).toDouble * a(i).toDouble
      normB += b(i).toDouble * b(i).toDouble
      i += 1
    }
    if (normA == 0.0 || normB == 0.0) 0.0f
    else (dot / (math.sqrt(normA) * math.sqrt(normB))).toFloat
  }

  /** Pairwise cosine similarity matrix for a set of vectors. The diagonal is always `1.0`. */
  def similarityMatrix(vectors: Array[Array[Float]]): Array[Array[Float]] = {
    val n = vectors.length
    val matrix = Array.fill(n, n)(0.0f)
    var i = 0
    while (i < n) {
      matrix(i)(i) = 1.0f
      var j = i + 1
      while (j < n) {
        val sim = cosineSimilarity(vectors(i), vectors(j))
        matrix(i)(j) = sim
        matrix(j)(i) = sim
        j += 1
      }
      i += 1
    }
    matrix
  }

  /** Partitions `n` samples into clusters using single-link agglomerative clustering: samples `i`
    * and `j` end up in the same cluster iff there is a chain of pairwise similarities, each `>=
    * threshold`, connecting them.
    *
    * @return
    *   for each sample index, the smallest sample index in its cluster - a canonical, order-
    *   independent cluster id.
    */
  def clusterBySimilarity(similarity: Array[Array[Float]], threshold: Float): Array[Int] = {
    val n = similarity.length
    val parent = Array.tabulate(n)(identity)

    def find(i: Int): Int = {
      if (parent(i) == i) i
      else {
        val root = find(parent(i))
        parent(i) = root
        root
      }
    }

    def union(i: Int, j: Int): Unit = {
      val ri = find(i)
      val rj = find(j)
      if (ri != rj) parent(math.max(ri, rj)) = math.min(ri, rj)
    }

    var i = 0
    while (i < n) {
      var j = i + 1
      while (j < n) {
        if (similarity(i)(j) >= threshold) union(i, j)
        j += 1
      }
      i += 1
    }

    Array.tabulate(n)(find)
  }

  /** Groups sample indexes by their cluster id (as returned by [[clusterBySimilarity]]) into
    * clusters ordered by their canonical id, with members ordered by sample index - fully
    * deterministic regardless of computation order.
    */
  def groupByCluster(clusterIds: Array[Int]): Seq[Seq[Int]] = {
    clusterIds.zipWithIndex
      .groupBy(_._1)
      .toSeq
      .sortBy(_._1)
      .map { case (_, members) => members.map(_._2).sorted.toSeq }
  }

  // ---------------------------------------------------------------------------------------------
  // Semantic Entropy (Kuhn et al. 2023; matches TruthTorchLM's SemanticEntropy)
  // ---------------------------------------------------------------------------------------------

  private def logSumExp(values: Seq[Double]): Double = {
    val maxV = values.max
    if (maxV.isNegInfinity) Double.NegativeInfinity
    else maxV + math.log(values.map(v => math.exp(v - maxV)).sum)
  }

  /** Cluster-based semantic entropy: for each semantic cluster, accumulates the negative
    * log-sum-exp of the sample scores in that cluster (a numerically stable "log cluster mass"),
    * then averages over the number of clusters. This is the same estimator TruthTorchLM / Kuhn et
    * al. use, generalized over black-box (uniform-score) and white-box (logprob-weighted) inputs
    * via the `sequenceLogProbs` argument.
    *
    * @param clusterIds
    *   cluster assignment per sample, as returned by [[clusterBySimilarity]]
    * @param sequenceLogProbs
    *   per-sample length-normalized sequence log-likelihoods (i.e. the *mean* of each sample's
    *   token logprobs, not the raw sum - mixing normalized and unnormalized scores biases entropy
    *   towards long or short samples). When not supplied (the common black-box case, no per-token
    *   logprobs available), every sample is treated as equally likely.
    * @return
    *   the semantic entropy (higher = more uncertain); exactly `0.0` when there is only one
    *   cluster (all samples semantically equivalent)
    */
  def semanticEntropy(
      clusterIds: Array[Int],
      sequenceLogProbs: Option[Array[Double]] = None): Double = {
    val n = clusterIds.length
    require(n > 0, "clusterIds must not be empty")
    val scores = sequenceLogProbs.getOrElse(Array.fill(n)(-math.log(n.toDouble)))
    require(
      scores.length == n,
      s"sequenceLogProbs must have exactly one entry per sample, got ${scores.length} for $n " +
        "samples")

    val clusters = groupByCluster(clusterIds)
    if (clusters.length <= 1) 0.0
    else {
      val perClusterNegLogMass = clusters.map(members => -logSumExp(members.map(scores)))
      perClusterNegLogMass.sum / clusters.length
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Eccentricity (Lin, Trivedi & Sun 2023 "Generating with Confidence"; matches TruthTorchLM's
  // calculate_U_ecc/calculate_C_ecc)
  // ---------------------------------------------------------------------------------------------

  /** Symmetric normalized graph Laplacian `D^-1/2 (D - W) D^-1/2` of an affinity matrix `w`,
    * where `D` is the diagonal degree matrix (row sums of `w`).
    */
  private def normalizedLaplacian(affinityMatrix: Array[Array[Double]]): DenseMatrix[Double] = {
    val degrees = affinityMatrix.map(_.sum)
    val w = DenseMatrix(affinityMatrix: _*)
    val dInvSqrt = diag(DenseVector(degrees.map(d => if (d > 0) 1.0 / math.sqrt(d) else 0.0)))
    val d = diag(DenseVector(degrees))
    dInvSqrt * (d - w) * dInvSqrt
  }

  /** Eccentricity uncertainty/confidence: embeds each sample using the eigenvectors of the
    * affinity matrix's normalized graph Laplacian whose eigenvalues are below `eigenThreshold`,
    * then scores each sample by its Euclidean distance from the centroid of that embedding. Small
    * eigenvalues of the normalized Laplacian correspond to well-separated semantic clusters, so
    * samples far from the centroid in this embedding are the ones that stand apart from the
    * consensus answer.
    *
    * Negative affinities are clamped to `0`. Spectral clustering on a graph Laplacian assumes
    * non-negative edge weights - a negative one makes a row's degree (its sum) shrink or go
    * negative, which is not a meaningful "how connected is this sample" quantity and lets the `d
    * > 0` guard in [[normalizedLaplacian]] silently zero out a whole row. The NLI backend is
    * already non-negative (probabilities), but cosine similarity of sentence embeddings ranges
    * over `[-1, 1]`, and two samples that disagree are no less "unrelated" than two that are
    * merely orthogonal.
    *
    * @param affinityMatrix
    *   n x n symmetric similarity matrix with `1.0` on the diagonal, e.g. from
    *   [[similarityMatrix]]
    * @param eigenThreshold
    *   keep only eigenvectors whose eigenvalue is strictly below this value (default `0.9`,
    *   matching the reference implementation)
    * @return
    *   (aggregate uncertainty = sum of per-sample scores, per-sample scores); both are exactly
    *   `0.0` when `n == 1` or when all samples are mutually identical
    */
  def eccentricity(
      affinityMatrix: Array[Array[Double]],
      eigenThreshold: Double = 0.9): (Double, Array[Double]) = {
    val n = affinityMatrix.length
    require(n > 0, "affinityMatrix must not be empty")
    if (n == 1) return (0.0, Array(0.0))

    val laplacian = normalizedLaplacian(affinityMatrix.map(_.map(math.max(_, 0.0))))
    val eigSym.EigSym(eigenvalues, eigenvectors) = eigSym(laplacian)

    val keptCols = (0 until n).filter(i => eigenvalues(i) < eigenThreshold)
    // The normalized Laplacian always has an exact (up to floating point) zero eigenvalue, so
    // keptCols is never empty in practice; this is a defensive fallback, not a real code path.
    val cols = if (keptCols.isEmpty) Seq(0) else keptCols

    val embedding: Array[Array[Double]] =
      Array.tabulate(n, cols.length) { case (row, col) => eigenvectors(row, cols(col)) }

    val centroid: Array[Double] =
      Array.tabulate(cols.length)(col => embedding.map(_(col)).sum / n)

    val perSample: Array[Double] = embedding.map { row =>
      math.sqrt(row.indices.map(col => math.pow(row(col) - centroid(col), 2)).sum)
    }

    (perSample.sum, perSample)
  }

  // ---------------------------------------------------------------------------------------------
  // White-box parsing: completion_probabilities (see AutoGGUFModel's outputLogProbs)
  // ---------------------------------------------------------------------------------------------

  private implicit val jsonFormats: Formats = DefaultFormats

  /** Length-normalized mean per-token log-likelihood of a completion, parsed from the verbatim
    * `completion_probabilities` JSON that `AutoGGUFModel` writes to annotation metadata when
    * `outputLogProbs` is enabled (an array of `{"logprob": ..., "top_logprobs": [...], ...}`
    * objects, one per generated token). This is both a white-box uncertainty score in its own
    * right (see `perplexity` = `exp(-meanLogProb)`) and the `sequenceLogProbs` input
    * `semanticEntropy` expects.
    *
    * Only the top-level `logprob` of each generated token is used - deliberately NOT a recursive
    * search, since each token's `top_logprobs` also nests a `logprob` field for every alternative
    * candidate token, which would silently pull in probabilities of tokens that were never
    * actually generated.
    *
    * @return
    *   `None` if `json` does not parse to a non-empty array of objects containing a numeric
    *   top-level `logprob` field
    */
  def meanLogProb(json: String): Option[Double] = {
    try {
      val tokenLogProbs = parse(json) match {
        case JArray(tokens) =>
          tokens.flatMap { token =>
            (token \ "logprob").extractOpt[Double]
          }
        case _ => Seq.empty
      }
      if (tokenLogProbs.isEmpty) None else Some(tokenLogProbs.sum / tokenLogProbs.length)
    } catch {
      case _: Exception => None
    }
  }

  /** Predictive entropy, averaged over generated tokens, parsed from the verbatim
    * `completion_probabilities` JSON (requires `AutoGGUFModel.setNProbs(k)` with `k > 1`, so each
    * token carries `top_logprobs` alternatives). Per token, entropy is computed over just the
    * observed top-k alternatives (`H = -sum(p_i * log(p_i))`, `p_i = exp(logprob_i)`) - this
    * necessarily *underestimates* true predictive entropy since it ignores probability mass
    * outside the top k, but is the best signal recoverable from a top-k-truncated distribution.
    *
    * @return
    *   `None` if `json` does not parse, is empty, or no token has a `top_logprobs` array with at
    *   least 2 entries (i.e. `nProbs` was not set to more than 1)
    */
  def predictiveEntropy(json: String): Option[Double] = {
    try {
      val perTokenEntropy = parse(json) match {
        case JArray(tokens) =>
          tokens.flatMap { token =>
            (token \ "top_logprobs").extractOpt[Seq[JValue]].flatMap { alternatives =>
              val logProbs = alternatives.flatMap(alt => (alt \ "logprob").extractOpt[Double])
              if (logProbs.length < 2) None
              else Some(-logProbs.map(lp => math.exp(lp) * lp).sum)
            }
          }
        case _ => Seq.empty
      }
      if (perTokenEntropy.isEmpty) None
      else Some(perTokenEntropy.sum / perTokenEntropy.length)
    } catch {
      case _: Exception => None
    }
  }

  /** Parses the `entailment_matrix` metadata JSON written by `SampleEntailmentMatrix` (a
    * row-major N x N array of entailment probabilities) back into a matrix.
    *
    * @return
    *   `None` if `json` does not parse to a well-formed square matrix of numbers
    */
  def parseEntailmentMatrix(json: String): Option[Array[Array[Float]]] = {
    try {
      parse(json) match {
        case JArray(rows) =>
          val matrix = rows.map {
            case JArray(cols) => cols.flatMap(_.extractOpt[Double].map(_.toFloat)).toArray
            case _ => Array.empty[Float]
          }.toArray
          if (matrix.nonEmpty && matrix.forall(_.length == matrix.length)) Some(matrix) else None
        case _ => None
      }
    } catch {
      case _: Exception => None
    }
  }

  /** Parses the `token_importance` metadata JSON written by `MarsTokenImportance` (an array of
    * `{"begin", "end", "importance"}` objects, with `end` inclusive, per the `Annotation`
    * convention this codebase uses everywhere) into `(begin, end, importance)` tuples, as
    * [[alignByCharSpan]] expects for its `phrases` argument.
    */
  def parseTokenImportance(json: String): Seq[(Int, Int, Double)] = {
    try {
      parse(json) match {
        case JArray(phrases) =>
          phrases.flatMap { phrase =>
            for {
              begin <- (phrase \ "begin").extractOpt[Int]
              end <- (phrase \ "end").extractOpt[Int]
              importance <- (phrase \ "importance").extractOpt[Double]
            } yield (begin, end, importance)
          }
        case _ => Seq.empty
      }
    } catch {
      case _: Exception => Seq.empty
    }
  }

  /** One generated token's character span (into the completion text) and log-likelihood, parsed
    * from `completion_probabilities`. `begin` is inclusive, `end` exclusive.
    */
  type TokenSpan = CompletionProbabilities.TokenSpan
  val TokenSpan: CompletionProbabilities.TokenSpan.type = CompletionProbabilities.TokenSpan

  /** Per-token character spans and logprobs, parsed from the verbatim `completion_probabilities`
    * JSON - see [[CompletionProbabilities.tokenSpans]], which owns this format because
    * `AutoGGUFModel` has to keep the array in sync with the completion text it post-processes.
    *
    * @return
    *   empty if `json` does not parse to a non-empty array of objects containing numeric
    *   `logprob` and integer-array `bytes` fields
    */
  def tokenSpansFromCompletionProbabilities(json: String): Seq[TokenSpan] =
    CompletionProbabilities.tokenSpans(json)

  /** Redistributes MARS phrase importance scores onto the generating LLM's own tokens by
    * character-span overlap, so importance (from a BERT tokenization of the answer) and
    * log-likelihood (from the generating model's own tokenization) end up on a common, exactly
    * character-aligned grid without any string matching.
    *
    * This deliberately replaces TruthTorchLM's reference approach (greedy fuzzy string matching
    * between decoded token text) with an exact character-overlap join: both `llmTokens`'
    * `completion_probabilities` and `phrases`' wordpiece offsets (`TokenPiece.begin/end`) carry
    * real character spans into the same completion text, so overlap is well-defined and
    * deterministic. A token that overlaps no phrase gets importance `0.0`.
    *
    * The two span conventions differ and are reconciled here: LLM token spans are half-open
    * (`end` exclusive, as [[CompletionProbabilities.tokenSpans]] produces them), while MARS
    * phrase spans come from `TokenPiece.begin`/`end`, which are inclusive like every other
    * `Annotation` offset in this codebase. Treating a phrase's inclusive `end` as if it were
    * exclusive would drop the last character of every phrase from the overlap, so a token
    * covering exactly that character would come back with importance `0.0`.
    *
    * @param llmTokens
    *   the generating model's tokens, in order, as (begin, endExclusive, negLogLikelihood)
    * @param phrases
    *   MARS phrases, as (begin, endInclusive, importance)
    * @return
    *   one (negLogLikelihood, importance) pair per LLM token, in `llmTokens` order
    */
  def alignByCharSpan(
      llmTokens: Seq[(Int, Int, Double)],
      phrases: Seq[(Int, Int, Double)]): Seq[(Double, Double)] = {
    llmTokens.map { case (tokenBegin, tokenEnd, negLogLikelihood) =>
      val overlaps = phrases.flatMap { case (phraseBegin, phraseEndInclusive, importance) =>
        val overlapLength =
          math.min(tokenEnd, phraseEndInclusive + 1) - math.max(tokenBegin, phraseBegin)
        if (overlapLength > 0) Some((overlapLength.toDouble, importance)) else None
      }
      val importance =
        if (overlaps.isEmpty) 0.0
        else overlaps.map { case (weight, imp) => weight * imp }.sum / overlaps.map(_._1).sum
      (negLogLikelihood, importance)
    }
  }

  private def softmax(values: Seq[Double]): Seq[Double] = {
    val maxV = values.max
    val expValues = values.map(v => math.exp(v - maxV))
    val total = expValues.sum
    expValues.map(_ / total)
  }

  /** MARS (Bakman et al. 2024) uncertainty score: softmax-normalizes the per-token importance
    * weights (after dividing by `temperature`), then combines the importance-weighted mean
    * negative log-likelihood with the plain mean negative log-likelihood in equal parts. Matches
    * TruthTorchLM's `compute_token_nll_importance_phrase`, oriented as uncertainty (higher = less
    * certain) rather than their `truth_value = -score`.
    *
    * @param nllAndImportance
    *   one (negLogLikelihood, importance) pair per LLM token, as returned by [[alignByCharSpan]]
    * @param temperature
    *   softmax temperature applied to the importance weights before normalizing (Default `0.1`,
    *   matching the reference implementation)
    * @return
    *   the MARS uncertainty score; `0.0` for an empty sequence
    */
  def marsScore(nllAndImportance: Seq[(Double, Double)], temperature: Double = 0.1): Double = {
    if (nllAndImportance.isEmpty) return 0.0
    val (negLogLikelihoods, rawImportance) = nllAndImportance.unzip
    val normalizedImportance = softmax(rawImportance.map(_ / temperature))
    val importanceWeightedNll =
      normalizedImportance.zip(negLogLikelihoods).map { case (w, nll) => w * nll }.sum
    0.5 * importanceWeightedNll + 0.5 * (negLogLikelihoods.sum / negLogLikelihoods.length)
  }
}
