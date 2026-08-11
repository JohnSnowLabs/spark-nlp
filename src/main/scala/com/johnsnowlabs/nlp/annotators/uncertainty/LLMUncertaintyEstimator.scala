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

import com.johnsnowlabs.nlp._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

/** Estimates how uncertain an LLM is about a completion it generated, from one or more sampled
  * completions.
  *
  * This annotator computes no logits and loads no model of its own: it is a pure post-processor
  * over annotations produced upstream in the pipeline. Two families of methods are supported (see
  * [[https://arxiv.org/abs/2506.01114 Bakman et al. 2025]] for why these specific methods were
  * chosen: they are the only ones the paper found to keep low error across calibration-set
  * distribution shift):
  *
  *   - '''Black box''' (`semanticEntropy`, `eccentricity`): needs multiple sampled completions
  *     for the same prompt (`AutoGGUFModel.setNumSamples(n)`) plus a way to tell which samples
  *     mean the same thing - either the default `similarityBackend="embeddings"` (cosine
  *     similarity of an additional sentence-embeddings input column, e.g. from
  *     [[https://sparknlp.org MPNetEmbeddings]] or `MiniLMEmbeddings` - both Sentence-BERT
  *     models trained on NLI/STS data for exactly this "are these two answers equivalent" task,
  *     unlike retrieval-oriented embedders such as E5 or BGE - alongside the completions
  *     column), or `similarityBackend="nli"` (bidirectional entailment, faithful to the original
  *     Semantic Entropy paper - needs a `SampleEntailmentMatrix` stage run over the same
  *     completions column beforehand instead of an embeddings column).
  *   - '''White box''' (`mars`, `meanLogProb`, `perplexity`, `predictiveEntropy`): needs
  *     per-token log probabilities (`AutoGGUFModel.setOutputLogProbs(true)`, and for
  *     `predictiveEntropy` also `setNProbs(k > 1)`). `mars` additionally needs a
  *     `MarsTokenImportance` stage run over the same completions column beforehand. Unlike the
  *     black-box methods, these work with a single sample (`numSamples` does not need to be `>
  *     1`) and cost about as much as one generation, since no resampling is needed.
  *
  * ==Score direction==
  * `uncertainty_score` is oriented so that '''higher means more uncertain'''; `confidence_score`
  * is `1 - uncertainty_score` (both normalized to roughly `[0, 1]`, see `ensemble` below for the
  * normalization caveat). This holds for every individual method's raw score too (e.g.
  * `semantic_entropy`, `eccentricity`, `perplexity`, `mars` in the output metadata) -
  * '''except''' `mean_log_prob`, which is stored under its natural, confidence-oriented meaning
  * (closer to `0` \= more confident, as a log-probability should read) and only sign-flipped
  * internally when folded into `uncertainty_score`.
  *
  * ==Calibration caveat==
  * A raw uncertainty score on its own does not tell you whether an answer should be trusted: the
  * paper this annotator is based on found that decision thresholds must be calibrated on data
  * resembling your deployment distribution, or error rates rise sharply. Set the `threshold`
  * param (once you have calibrated one on your own data) to get a boolean `is_reliable` metadata
  * flag; without it, only the raw score is emitted.
  *
  * ==Reasoning-model caveat==
  * For black-box methods, embed and cluster the final answer only - not the raw completion.
  * Reasoning models (e.g. Qwen3 with thinking enabled) prepend a `<think>...</think>` block to
  * every completion; if that block is left in, its boilerplate phrasing ("Okay, the user asked
  * for X. Let me think...") tends to be near-identical across samples even when the actual
  * answers differ substantially, which drowns out the real signal and collapses clustering
  * towards a single, spuriously low-uncertainty cluster. Strip it with
  * `AutoGGUFModel.setRemoveThinkingTag("think")` before it reaches this annotator's input column.
  *
  * ==Validated==
  * Every method (`semanticEntropy` under both backends, `eccentricity`, `mars`, and the
  * `meanLogProb`/`perplexity`/`predictiveEntropy` family, plus `ensemble`) has been run
  * end-to-end against real sampled completions and shown to genuinely separate wrong answers
  * from right ones - each one scores clearly above the random-guessing baseline. This is
  * directional evidence from a small internal QA benchmark on one model, not a calibrated,
  * generalizable accuracy claim - see the Calibration caveat above before trusting a threshold
  * on your own data.
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  *
  * val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
  *
  * val llm = AutoGGUFModel.pretrained()
  *   .setInputCols("document").setOutputCol("completions")
  *   .setNumSamples(5).setTemperature(0.7f)
  *
  * val embeddings = MPNetEmbeddings.pretrained()
  *   .setInputCols("completions").setOutputCol("sample_embeddings")
  *
  * val uncertainty = new LLMUncertaintyEstimator()
  *   .setInputCols("completions", "sample_embeddings").setOutputCol("uncertainty")
  *   .setMethods(Array("semanticEntropy"))
  *
  * val pipeline = new Pipeline().setStages(Array(document, llm, embeddings, uncertainty))
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio setParam  3
  * @groupprio getParam  4
  */
class LLMUncertaintyEstimator(override val uid: String)
    extends AnnotatorModel[LLMUncertaintyEstimator]
    with HasSimpleAnnotate[LLMUncertaintyEstimator] {

  def this() = this(Identifiable.randomUID("LLM_UNCERTAINTY_ESTIMATOR"))

  /** Input Annotator Type: DOCUMENT (the sampled completions, e.g. from `AutoGGUFModel`)
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(AnnotatorType.DOCUMENT)

  /** Optional input Annotator Type: SENTENCE_EMBEDDINGS (required when `similarityBackend` is
    * `"embeddings"`, the default, and `methods` includes `semanticEntropy` or `eccentricity`)
    *
    * @group anno
    */
  override val optionalInputAnnotatorTypes: Array[AnnotatorType] =
    Array(AnnotatorType.SENTENCE_EMBEDDINGS)

  /** Output Annotator Type: DOCUMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.DOCUMENT

  /** Which uncertainty method(s) to compute (Default: `Array("semanticEntropy")`). One or more
    * of: `semanticEntropy`, `eccentricity` (black box; need multiple sampled completions),
    * `mars`, `meanLogProb`, `perplexity`, `predictiveEntropy` (white box).
    *
    * @group param
    */
  val methods = new StringArrayParam(this, "methods", "Uncertainty method(s) to compute")

  /** How to determine which sampled completions mean the same thing for the black-box methods
    * (Default: `"embeddings"`). `"embeddings"` clusters by cosine similarity of a sentence-
    * embeddings input column; `"nli"` clusters by bidirectional entailment, using
    * `entailment_matrix` metadata from a `SampleEntailmentMatrix` stage run over the same
    * completions column beforehand.
    *
    * @group param
    */
  val similarityBackend =
    new Param[String](this, "similarityBackend", "'embeddings' or 'nli'")

  /** Cosine-similarity threshold above which two sampled completions are considered to belong to
    * the same semantic cluster, used by `semanticEntropy` with `similarityBackend='embeddings'`
    * (Default: `0.85`).
    *
    * @group param
    */
  val similarityThreshold = new FloatParam(
    this,
    "similarityThreshold",
    "Cosine similarity threshold for clustering samples into semantic equivalence classes")

  /** Entailment-probability threshold above which two sampled completions are considered to
    * belong to the same semantic cluster, used by `semanticEntropy` with
    * `similarityBackend='nli'` (Default: `0.5`).
    *
    * @group param
    */
  val entailmentThreshold = new FloatParam(
    this,
    "entailmentThreshold",
    "Entailment probability threshold for the NLI similarity backend")

  /** Keep only eigenvectors of the normalized graph Laplacian whose eigenvalue is below this
    * value, used by `eccentricity` (Default: `0.9`, matching the reference implementation).
    *
    * @group param
    */
  val eigenThreshold = new FloatParam(
    this,
    "eigenThreshold",
    "Eigenvalue cutoff for the eccentricity method's spectral embedding")

  /** Whether to combine multiple requested `methods` into a single ensembled `uncertainty_score`
    * (Default: `false`, meaning the first method in `methods` is used as the primary score).
    * Ensembling averages each method's score after a rough per-method normalization (documented
    * per-method below) - without calibration on your own data this remains approximate.
    *
    * @group param
    */
  val ensemble = new BooleanParam(
    this,
    "ensemble",
    "Whether to combine multiple methods into a single ensembled uncertainty score")

  /** Positional weights (matching `methods`) used when `ensemble` is `true` (Default: equal
    * weights).
    *
    * @group param
    */
  val ensembleWeights =
    new DoubleArrayParam(this, "ensembleWeights", "Positional per-method weights for ensembling")

  /** A calibrated uncertainty threshold: when set, an `is_reliable` metadata flag (`"true"` when
    * the (ensembled or primary) uncertainty score is at or below this value) is added to the
    * output. Unset by default - see the class-level calibration caveat.
    *
    * @group param
    */
  val threshold = new DoubleParam(
    this,
    "threshold",
    "Calibrated uncertainty threshold; when set, adds an is_reliable metadata flag")

  /** @group setParam */
  def setMethods(methods: Array[String]): this.type = {
    require(methods.nonEmpty, "methods must not be empty")
    val unknown = methods.toSet -- LLMUncertaintyEstimator.supportedMethods
    require(
      unknown.isEmpty,
      s"Unknown uncertainty method(s): ${unknown.mkString(", ")}. Supported: " +
        LLMUncertaintyEstimator.supportedMethods.mkString(", "))
    set(this.methods, methods)
  }

  /** @group setParam */
  def setSimilarityBackend(similarityBackend: String): this.type = {
    require(
      Set("embeddings", "nli").contains(similarityBackend),
      s"similarityBackend must be 'embeddings' or 'nli', got '$similarityBackend'")
    set(this.similarityBackend, similarityBackend)
  }

  /** @group setParam */
  def setSimilarityThreshold(similarityThreshold: Float): this.type = {
    set(this.similarityThreshold, similarityThreshold)
  }

  /** @group setParam */
  def setEntailmentThreshold(entailmentThreshold: Float): this.type = {
    set(this.entailmentThreshold, entailmentThreshold)
  }

  /** @group setParam */
  def setEigenThreshold(eigenThreshold: Float): this.type = {
    set(this.eigenThreshold, eigenThreshold)
  }

  /** @group setParam */
  def setEnsemble(ensemble: Boolean): this.type = { set(this.ensemble, ensemble) }

  /** @group setParam */
  def setEnsembleWeights(ensembleWeights: Array[Double]): this.type = {
    set(this.ensembleWeights, ensembleWeights)
  }

  /** @group setParam */
  def setThreshold(threshold: Double): this.type = { set(this.threshold, threshold) }

  /** @group getParam */
  def getMethods: Array[String] = $(methods)

  /** @group getParam */
  def getSimilarityBackend: String = $(similarityBackend)

  /** @group getParam */
  def getSimilarityThreshold: Float = $(similarityThreshold)

  /** @group getParam */
  def getEntailmentThreshold: Float = $(entailmentThreshold)

  /** @group getParam */
  def getEigenThreshold: Float = $(eigenThreshold)

  /** @group getParam */
  def getEnsemble: Boolean = $(ensemble)

  /** Positional per-method ensemble weights, or `None` when unset (meaning equal weights).
    *
    * @group getParam
    */
  def getEnsembleWeights: Option[Array[Double]] = get(ensembleWeights)

  /** The calibrated uncertainty threshold, or `None` when unset (meaning no `is_reliable` flag is
    * emitted).
    *
    * @group getParam
    */
  def getThreshold: Option[Double] = get(threshold)

  setDefault(
    methods -> Array("semanticEntropy"),
    similarityBackend -> "embeddings",
    similarityThreshold -> 0.85f,
    entailmentThreshold -> 0.5f,
    eigenThreshold -> 0.9f,
    ensemble -> false)

  /** Validates param combinations that individual setters cannot check on their own, on the
    * driver, before any row is processed.
    *
    * These checks are not redundant with the `require`s in `setMethods` / `setSimilarityBackend`:
    * pyspark's `_transfer_params_to_java` writes the raw `Param` values straight onto the Java
    * object, so nothing set from Python ever goes through a Scala setter. Without a driver-side
    * check, a typo'd method name from Python would first surface as a per-row exception on an
    * executor, deep inside `annotate`.
    */
  private[uncertainty] def validateParams(): Unit = {
    val configured = getMethods
    require(configured.nonEmpty, s"$uid: methods must not be empty")
    val unknown = configured.toSet -- LLMUncertaintyEstimator.supportedMethods
    require(
      unknown.isEmpty,
      s"$uid: unknown uncertainty method(s): ${unknown.toSeq.sorted.mkString(", ")}. Supported: " +
        LLMUncertaintyEstimator.supportedMethods.toSeq.sorted.mkString(", "))
    require(
      configured.distinct.length == configured.length,
      s"$uid: methods must not repeat, got ${configured.mkString(", ")}. Ensemble weights are " +
        "positional, so a repeated method would make the mapping between methods and weights " +
        "ambiguous.")
    require(
      Set("embeddings", "nli").contains(getSimilarityBackend),
      s"$uid: similarityBackend must be 'embeddings' or 'nli', got '$getSimilarityBackend'")

    getEnsembleWeights.foreach { weights =>
      require(
        weights.length == configured.length,
        s"$uid: ensembleWeights has ${weights.length} entries but methods has " +
          s"${configured.length} (${configured.mkString(", ")}). Weights are positional, so " +
          "there must be exactly one per method.")
      require(
        weights.forall(_ >= 0.0),
        s"$uid: ensembleWeights must all be non-negative, got ${weights.mkString(", ")}")
      require(
        weights.sum > 0.0,
        s"$uid: ensembleWeights must not sum to zero, got ${weights.mkString(", ")}")
    }
  }

  override protected def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    validateParams()
    dataset
  }

  override protected def extraValidateMsg: String =
    s"$uid: methods ${getMethods.filter(LLMUncertaintyEstimator.blackBoxMethods.contains).mkString(", ")} " +
      s"with similarityBackend='embeddings' require a ${AnnotatorType.SENTENCE_EMBEDDINGS} input " +
      "column in addition to the completions column, e.g. " +
      ".setInputCols(\"completions\", \"sample_embeddings\")"

  override protected def extraValidate(schema: StructType): Boolean = {
    val needsEmbeddings =
      getSimilarityBackend == "embeddings" &&
        getMethods.exists(LLMUncertaintyEstimator.blackBoxMethods.contains)
    !needsEmbeddings || checkSchema(schema, AnnotatorType.SENTENCE_EMBEDDINGS)
  }

  private def requireEmbeddings(
      completions: Seq[Annotation],
      embeddingAnnotations: Seq[Annotation]): Array[Array[Float]] = {
    val vectors = embeddingAnnotations.map(_.embeddings).toArray
    require(
      vectors.length == completions.length,
      s"LLMUncertaintyEstimator with similarityBackend='embeddings' requires one " +
        s"${AnnotatorType.SENTENCE_EMBEDDINGS} annotation per sampled completion; got " +
        s"${vectors.length} embeddings for ${completions.length} samples. Pass a sentence-" +
        "embeddings column (e.g. from MPNetEmbeddings, computed over the same completions column) " +
        "as an additional input column.")
    vectors
  }

  /** Row-major N x N entailment matrix, symmetrized as `(entail(i,j) + entail(j,i)) / 2` (this is
    * `calculate_affinity_matrix`'s exact averaging formula in the reference implementation) with
    * `1.0` forced on the diagonal, parsed from `SampleEntailmentMatrix`'s `entailment_matrix`
    * metadata. The matrix describes the row as a whole and is written once per row, so it is
    * looked up across the row's completions rather than assumed to sit on any particular one.
    */
  private def requireEntailmentMatrix(completions: Seq[Annotation]): Array[Array[Float]] = {
    val parsed = completions
      .flatMap(_.metadata.get("entailment_matrix"))
      .headOption
      .flatMap(UncertaintyMetrics.parseEntailmentMatrix)
    require(
      parsed.exists(_.length == completions.length),
      s"$uid: similarityBackend='nli' requires entailment_matrix metadata on the completions " +
        "column (run SampleEntailmentMatrix over the same completions column upstream), sized " +
        s"for all ${completions.length} sampled completions; got " +
        s"${parsed.map(_.length.toString).getOrElse("none")}.")
    val raw = parsed.get
    Array.tabulate(completions.length, completions.length) { (i, j) =>
      if (i == j) 1.0f else (raw(i)(j) + raw(j)(i)) / 2.0f
    }
  }

  /** Similarity matrix used for clustering (`semanticEntropy`) and the spectral embedding
    * (`eccentricity`), from whichever `similarityBackend` is configured.
    */
  private def similarityMatrixForClustering(
      completions: Seq[Annotation],
      embeddingAnnotations: Seq[Annotation]): Array[Array[Float]] = {
    getSimilarityBackend match {
      case "embeddings" =>
        UncertaintyMetrics.similarityMatrix(requireEmbeddings(completions, embeddingAnnotations))
      case "nli" =>
        requireEntailmentMatrix(completions)
      case other =>
        // Unreachable in practice: setSimilarityBackend already validates this.
        throw new IllegalArgumentException(s"Unknown similarityBackend: $other")
    }
  }

  private def clusteringThreshold: Float =
    if (getSimilarityBackend == "nli") getEntailmentThreshold else getSimilarityThreshold

  /** Computes one method's raw score plus any method-specific extra metadata (e.g.
    * `num_semantic_clusters` for `semanticEntropy`).
    */
  private def computeMethod(
      method: String,
      completions: Seq[Annotation],
      embeddingAnnotations: Seq[Annotation]): (Double, Map[String, String]) = {
    method match {
      case "semanticEntropy" =>
        val similarity = similarityMatrixForClustering(completions, embeddingAnnotations)
        val clusterIds = UncertaintyMetrics.clusterBySimilarity(similarity, clusteringThreshold)
        val score = UncertaintyMetrics.semanticEntropy(clusterIds)
        (score, Map("num_semantic_clusters" -> clusterIds.distinct.length.toString))
      case "eccentricity" =>
        val similarity =
          similarityMatrixForClustering(completions, embeddingAnnotations).map(_.map(_.toDouble))
        val (score, _) = UncertaintyMetrics.eccentricity(similarity, getEigenThreshold)
        (score, Map.empty)
      case "meanLogProb" =>
        val perSample =
          requireCompletionProbabilities(completions, method, UncertaintyMetrics.meanLogProb)
        // meanLogProb is confidence-oriented (closer to 0 = more confident); stored as-is since
        // that's what the name means, sign-flipped separately when folded into uncertainty_score.
        (perSample.sum / perSample.length, Map.empty)
      case "perplexity" =>
        val perSample =
          requireCompletionProbabilities(completions, method, UncertaintyMetrics.meanLogProb)
        val meanLp = perSample.sum / perSample.length
        (math.exp(-meanLp), Map.empty)
      case "predictiveEntropy" =>
        val perSample =
          requireCompletionProbabilities(
            completions,
            method,
            UncertaintyMetrics.predictiveEntropy)
        (perSample.sum / perSample.length, Map.empty)
      case "mars" =>
        val perSample = completions.flatMap { c =>
          for {
            probsJson <- c.metadata.get("completion_probabilities")
            importanceJson <- c.metadata.get("token_importance")
          } yield {
            val tokenSpans = UncertaintyMetrics.tokenSpansFromCompletionProbabilities(probsJson)
            val phrases = UncertaintyMetrics.parseTokenImportance(importanceJson)
            val aligned =
              UncertaintyMetrics.alignByCharSpan(
                tokenSpans.map(t => (t.begin, t.end, -t.logProb)),
                phrases)
            UncertaintyMetrics.marsScore(aligned)
          }
        }
        require(
          perSample.nonEmpty,
          s"$uid: method 'mars' requires both completion_probabilities metadata " +
            "(AutoGGUFModel.setOutputLogProbs(true)) and token_importance metadata (from " +
            "MarsTokenImportance, run over the same completions column) on every sampled " +
            "completion.")
        (perSample.sum / perSample.length, Map.empty)
      case other =>
        // Unreachable in practice: setMethods already validates against supportedMethods.
        throw new IllegalArgumentException(s"Unknown uncertainty method: $other")
    }
  }

  private def requireCompletionProbabilities(
      completions: Seq[Annotation],
      method: String,
      extract: String => Option[Double]): Seq[Double] = {
    val perSample =
      completions.flatMap(_.metadata.get("completion_probabilities").flatMap(extract))
    require(
      perSample.nonEmpty,
      s"$uid: method '$method' requires completion_probabilities metadata on every sampled " +
        "completion (set AutoGGUFModel.setOutputLogProbs(true), and for predictiveEntropy also " +
        "setNProbs(k > 1), upstream).")
    perSample
  }

  /** Rough per-method normalizer so different methods can be averaged together in `ensemble`
    * mode. `semanticEntropy` has an exact maximum of `ln(numSamples)` (all samples in their own
    * cluster); `eccentricity` has no closed-form maximum, so `sqrt(numSamples)` is used as a
    * scale heuristic; the white-box methods (`meanLogProb`, `perplexity`, `predictiveEntropy`,
    * `mars`) have no natural bound at all without calibration data, so they are left unscaled -
    * `confidence_score` is clamped to `[0, 1]` regardless of how any of this normalizes.
    */
  private def normalizer(method: String, numSamples: Int): Double = method match {
    case "semanticEntropy" => if (numSamples > 1) math.log(numSamples.toDouble) else 1.0
    case "eccentricity" => math.sqrt(numSamples.toDouble)
    case _ => 1.0
  }

  /** The raw score's floor at zero uncertainty, i.e. what the *other* ensembled methods already
    * report for a perfectly confident sample. Only `perplexity` needs this: it's
    * `exp(-meanLogProb)`, mathematically bounded below by `1.0` (not `0.0`), so left as-is it
    * contributes a near-constant ~1.0+ term that dominates an unweighted average against methods
    * that actually start at `0`. Subtracted only for ensembling; the raw `metadata['perplexity']`
    * value keeps its standard definition.
    */
  private def baseline(method: String): Double = method match {
    case "perplexity" => 1.0
    case _ => 0.0
  }

  /** `meanLogProb` is naturally confidence-oriented (closer to `0` = more confident); every other
    * method's raw score already increases with uncertainty, per the class-level score-direction
    * convention.
    */
  private def orientationSign(method: String): Double =
    if (method == "meanLogProb") -1.0 else 1.0

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val completions = annotations.filter(_.annotatorType == AnnotatorType.DOCUMENT)
    if (completions.isEmpty) return Seq.empty

    val numSamples = completions.length
    val resultText = completions.head.result
    val end = math.max(resultText.length - 1, -1)
    // These are all per-sample (or, for entailment_matrix, row-wide but only meaningful
    // alongside the samples it was computed from); copying sample 0's value onto the aggregated
    // output annotation would be misleading, so they are excluded from pass-through.
    val baseMetadata =
      completions.head.metadata -- Set(
        "sample_index",
        "completion_probabilities",
        "token_importance",
        "entailment_matrix")

    if (completions.exists(_.metadata.contains("llamacpp_exception"))) {
      return Seq(
        new Annotation(
          outputAnnotatorType,
          0,
          end,
          resultText,
          baseMetadata + ("uncertainty_estimation_skipped" ->
            "upstream completion failed (llamacpp_exception present in input metadata)")))
    }

    val embeddingAnnotations =
      annotations.filter(_.annotatorType == AnnotatorType.SENTENCE_EMBEDDINGS)

    val warningMetadata: Map[String, String] =
      if (numSamples == 1)
        Map(
          "warning" ->
            ("Only one sample was provided; sample-based uncertainty methods (semanticEntropy, " +
              "eccentricity) report their trivial value (0.0 uncertainty). Set " +
              "AutoGGUFModel.setNumSamples(n > 1) upstream for a meaningful estimate."))
      else Map.empty

    val methodResults: Map[String, (Double, Map[String, String])] =
      getMethods.map(m => m -> computeMethod(m, completions, embeddingAnnotations)).toMap

    val scoreMetadata: Map[String, String] =
      methodResults.flatMap { case (method, (score, extra)) =>
        Map(LLMUncertaintyEstimator.metadataKeyFor(method) -> score.toString) ++ extra
      }

    val normalizedScores: Map[String, Double] = methodResults.map { case (method, (score, _)) =>
      method ->
        (orientationSign(method) * (score - baseline(method))) / normalizer(method, numSamples)
    }

    val primaryUncertainty: Double =
      if (getEnsemble && normalizedScores.size > 1) {
        // validateParams has already established one weight per method, all non-negative and not
        // summing to zero, so the positional lookup and the division below are both safe.
        val weights = getEnsembleWeights.getOrElse(Array.fill(getMethods.length)(1.0))
        val weighted = getMethods.zip(weights).map { case (m, w) => normalizedScores(m) * w }
        weighted.sum / weights.sum
      } else {
        normalizedScores(getMethods.head)
      }

    val confidence = 1.0 - math.min(math.max(primaryUncertainty, 0.0), 1.0)

    val summaryMetadata = Map(
      "uncertainty_score" -> primaryUncertainty.toString,
      "confidence_score" -> confidence.toString,
      "num_samples" -> numSamples.toString)

    val thresholdMetadata: Map[String, String] = getThreshold match {
      case Some(t) => Map("is_reliable" -> (primaryUncertainty <= t).toString)
      case None => Map.empty
    }

    Seq(
      new Annotation(
        outputAnnotatorType,
        0,
        end,
        resultText,
        baseMetadata ++ warningMetadata ++ scoreMetadata ++ summaryMetadata ++ thresholdMetadata))
  }
}

/** This is the companion object of [[LLMUncertaintyEstimator]]. Please refer to that class for
  * the documentation.
  */
object LLMUncertaintyEstimator extends DefaultParamsReadable[LLMUncertaintyEstimator] {

  private[uncertainty] val blackBoxMethods: Set[String] = Set("semanticEntropy", "eccentricity")
  private[uncertainty] val whiteBoxMethods: Set[String] =
    Set("mars", "meanLogProb", "perplexity", "predictiveEntropy")
  val supportedMethods: Set[String] = blackBoxMethods ++ whiteBoxMethods

  private[uncertainty] def metadataKeyFor(method: String): String = method match {
    case "semanticEntropy" => "semantic_entropy"
    case "predictiveEntropy" => "predictive_entropy"
    case "meanLogProb" => "mean_log_prob"
    case other => other
  }
}
