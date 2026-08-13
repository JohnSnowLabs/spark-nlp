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

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class LLMUncertaintyEstimatorTest extends AnyFlatSpec {

  // Forces a SparkSession to exist before the serialization test needs one.
  private val spark = ResourceHelper.spark

  private def completion(
      text: String,
      sampleIndex: Int,
      extraMetadata: Map[String, String] = Map.empty): Annotation =
    new Annotation(
      AnnotatorType.DOCUMENT,
      0,
      math.max(text.length - 1, -1),
      text,
      Map("sentence" -> "0", "sample_index" -> sampleIndex.toString) ++ extraMetadata)

  private def embedding(vector: Array[Float]): Annotation =
    new Annotation(AnnotatorType.SENTENCE_EMBEDDINGS, 0, 0, "", Map("sentence" -> "0"), vector)

  behavior of "LLMUncertaintyEstimator with similarityBackend=embeddings"

  it should "score three near-identical samples as more confident than three diverse ones" taggedAs FastTest in {
    val consensusEstimator = new LLMUncertaintyEstimator()
      .setInputCols("completions", "embeddings")
      .setOutputCol("uncertainty")
      .setMethods(Array("semanticEntropy"))

    val consensusInput = Seq(
      completion("Paris", 0),
      completion("Paris", 1),
      completion("Paris", 2),
      embedding(Array(1.0f, 0.0f)),
      embedding(Array(0.99f, 0.01f)),
      embedding(Array(0.98f, 0.02f)))
    val consensusResult = consensusEstimator.annotate(consensusInput)
    assert(consensusResult.length == 1)
    val consensusUncertainty = consensusResult.head.metadata("uncertainty_score").toDouble

    val diverseInput = Seq(
      completion("Paris", 0),
      completion("London", 1),
      completion("Berlin", 2),
      embedding(Array(1.0f, 0.0f)),
      embedding(Array(0.0f, 1.0f)),
      embedding(Array(-1.0f, 0.0f)))
    val diverseResult = consensusEstimator.annotate(diverseInput)
    val diverseUncertainty = diverseResult.head.metadata("uncertainty_score").toDouble

    assert(diverseUncertainty > consensusUncertainty)
  }

  it should "report num_semantic_clusters alongside semantic_entropy" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions", "embeddings")
      .setOutputCol("uncertainty")
      .setMethods(Array("semanticEntropy"))

    val input = Seq(
      completion("Paris", 0),
      completion("London", 1),
      embedding(Array(1.0f, 0.0f)),
      embedding(Array(0.0f, 1.0f)))
    val result = estimator.annotate(input)
    assert(result.head.metadata("num_semantic_clusters").toInt == 2)
  }

  it should "compute eccentricity without error and orient it as uncertainty" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions", "embeddings")
      .setOutputCol("uncertainty")
      .setMethods(Array("eccentricity"))

    val input = Seq(
      completion("Paris", 0),
      completion("Paris", 1),
      completion("London", 2),
      embedding(Array(1.0f, 0.0f)),
      embedding(Array(0.99f, 0.01f)),
      embedding(Array(0.0f, 1.0f)))
    val result = estimator.annotate(input)
    val uncertainty = result.head.metadata("uncertainty_score").toDouble
    assert(uncertainty > 0.0)
    val confidence = result.head.metadata("confidence_score").toDouble
    assert(confidence >= 0.0 && confidence <= 1.0)
  }

  behavior of "LLMUncertaintyEstimator degenerate cases"

  it should "emit a warning and trivial (0.0) uncertainty for a single sample" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions", "embeddings")
      .setOutputCol("uncertainty")
      .setMethods(Array("semanticEntropy"))

    val input = Seq(completion("Paris", 0), embedding(Array(1.0f, 0.0f)))
    val result = estimator.annotate(input)
    assert(result.length == 1)
    assert(result.head.metadata("uncertainty_score").toDouble == 0.0)
    assert(result.head.metadata.contains("warning"))
  }

  it should "skip scoring and pass through the exception when upstream completion failed" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions", "embeddings")
      .setOutputCol("uncertainty")
      .setMethods(Array("semanticEntropy"))

    val failedInput = Seq(
      new Annotation(
        AnnotatorType.DOCUMENT,
        0,
        -1,
        "",
        Map("sample_index" -> "0", "llamacpp_exception" -> "model failed to load")))
    val result = estimator.annotate(failedInput)
    assert(result.length == 1)
    assert(result.head.metadata.contains("uncertainty_estimation_skipped"))
    assert(!result.head.metadata.contains("uncertainty_score"))
  }

  it should "return empty output for empty input" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions", "embeddings")
      .setOutputCol("uncertainty")
    assert(estimator.annotate(Seq.empty).isEmpty)
  }

  it should "fail with a clear message when the embeddings backend is missing embeddings" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions", "embeddings")
      .setOutputCol("uncertainty")
      .setMethods(Array("semanticEntropy"))

    val input = Seq(completion("Paris", 0), completion("London", 1)) // no embeddings at all
    val ex = intercept[IllegalArgumentException] {
      estimator.annotate(input)
    }
    assert(ex.getMessage.contains(AnnotatorType.SENTENCE_EMBEDDINGS))
  }

  behavior of "LLMUncertaintyEstimator white-box methods"

  private def completionWithLogProbs(
      text: String,
      sampleIndex: Int,
      meanLogProb: Double): Annotation = {
    val json = s"""[{"logprob": $meanLogProb, "top_logprobs": []}]"""
    completion(text, sampleIndex, Map("completion_probabilities" -> json))
  }

  it should "compute meanLogProb, confidence-oriented in its own metadata" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions")
      .setOutputCol("uncertainty")
      .setMethods(Array("meanLogProb"))

    val input = Seq(completionWithLogProbs("Paris", 0, -0.5))
    val result = estimator.annotate(input)
    assert(result.head.metadata("mean_log_prob").toDouble == -0.5)
    // uncertainty_score should be the sign-flipped (positive) version
    assert(result.head.metadata("uncertainty_score").toDouble > 0.0)
  }

  it should "compute perplexity as exp(-meanLogProb)" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions")
      .setOutputCol("uncertainty")
      .setMethods(Array("perplexity"))

    val input = Seq(completionWithLogProbs("Paris", 0, -1.0))
    val result = estimator.annotate(input)
    val perplexity = result.head.metadata("perplexity").toDouble
    assert(math.abs(perplexity - math.exp(1.0)) < 1e-6)
  }

  it should "fail with a clear message when completion_probabilities metadata is missing" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions")
      .setOutputCol("uncertainty")
      .setMethods(Array("meanLogProb"))

    val ex = intercept[IllegalArgumentException] {
      estimator.annotate(Seq(completion("Paris", 0)))
    }
    assert(ex.getMessage.contains("completion_probabilities"))
  }

  it should "compute mars from completion_probabilities and token_importance together" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions")
      .setOutputCol("uncertainty")
      .setMethods(Array("mars"))

    val probsJson = """[{"logprob": -0.2, "bytes": [80, 97, 114, 105, 115]}]""" // "Paris"
    val importanceJson = """[{"begin": 0, "end": 5, "importance": 0.9}]"""
    val input = Seq(
      completion(
        "Paris",
        0,
        Map("completion_probabilities" -> probsJson, "token_importance" -> importanceJson)))
    val result = estimator.annotate(input)
    assert(result.head.metadata.contains("mars"))
  }

  behavior of "LLMUncertaintyEstimator ensembling"

  it should "average normalized scores across methods when ensemble=true" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions")
      .setOutputCol("uncertainty")
      .setMethods(Array("meanLogProb", "perplexity"))
      .setEnsemble(true)

    val input = Seq(completionWithLogProbs("Paris", 0, -1.0))
    val result = estimator.annotate(input)
    // Just confirm it doesn't throw and produces a defined score combining both methods.
    assert(result.head.metadata.contains("uncertainty_score"))
    assert(result.head.metadata.contains("mean_log_prob"))
    assert(result.head.metadata.contains("perplexity"))
  }

  it should "not let perplexity's nonzero floor skew the ensemble average for a maximally confident sample" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions")
      .setOutputCol("uncertainty")
      .setMethods(Array("meanLogProb", "perplexity"))
      .setEnsemble(true)

    // meanLogProb = 0.0 means every token was assigned probability 1.0 - maximal confidence.
    // perplexity = exp(-meanLogProb) sits at its mathematical floor of 1.0 here, not 0.0, so
    // without subtracting that floor before averaging, this sample's ensemble score came out
    // around 0.5 (meanLogProb's correct 0.0 averaged against perplexity's un-subtracted 1.0)
    // even though the model was completely certain.
    val input = Seq(completionWithLogProbs("Paris", 0, 0.0))
    val result = estimator.annotate(input)

    // The raw perplexity value itself keeps its standard definition; only its contribution to
    // the ensemble average is baseline-corrected.
    assert(math.abs(result.head.metadata("perplexity").toDouble - 1.0) < 1e-6)
    assert(result.head.metadata("uncertainty_score").toDouble < 0.05)
  }

  it should "still weigh perplexity into a higher ensemble score for a genuinely uncertain sample" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions")
      .setOutputCol("uncertainty")
      .setMethods(Array("meanLogProb", "perplexity"))
      .setEnsemble(true)

    val confidentScore = estimator
      .annotate(Seq(completionWithLogProbs("Paris", 0, 0.0)))
      .head
      .metadata("uncertainty_score")
      .toDouble
    val uncertainScore = estimator
      .annotate(Seq(completionWithLogProbs("Paris", 0, -2.0)))
      .head
      .metadata("uncertainty_score")
      .toDouble

    assert(uncertainScore > confidentScore)
  }

  behavior of "LLMUncertaintyEstimator threshold"

  it should "add is_reliable only when threshold is set" taggedAs FastTest in {
    val withoutThreshold = new LLMUncertaintyEstimator()
      .setInputCols("completions")
      .setOutputCol("uncertainty")
      .setMethods(Array("meanLogProb"))
    val withThreshold = new LLMUncertaintyEstimator()
      .setInputCols("completions")
      .setOutputCol("uncertainty")
      .setMethods(Array("meanLogProb"))
      .setThreshold(10.0)

    val input = Seq(completionWithLogProbs("Paris", 0, -0.5))
    assert(!withoutThreshold.annotate(input).head.metadata.contains("is_reliable"))
    assert(withThreshold.annotate(input).head.metadata.contains("is_reliable"))
  }

  behavior of "LLMUncertaintyEstimator serialization"

  it should "round-trip its parameters through write/load" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions", "embeddings")
      .setOutputCol("uncertainty")
      .setMethods(Array("eccentricity"))
      .setSimilarityThreshold(0.7f)
      .setEnsemble(true)
      .setThreshold(0.3)

    val savePath = "./tmp_llm_uncertainty_estimator"
    estimator.write.overwrite().save(savePath)
    val loaded = LLMUncertaintyEstimator.load(savePath)

    assert(loaded.getMethods.toSeq == Seq("eccentricity"))
    assert(math.abs(loaded.getSimilarityThreshold - 0.7f) < 1e-6)
    assert(loaded.getEnsemble)
    assert(loaded.getThreshold.exists(t => math.abs(t - 0.3) < 1e-6))
  }

  it should "round-trip ensembleWeights alongside methods" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions")
      .setOutputCol("uncertainty")
      .setMethods(Array("meanLogProb", "perplexity"))
      .setEnsemble(true)
      .setEnsembleWeights(Array(0.25, 0.75))

    val savePath = "./tmp_llm_uncertainty_estimator_weights"
    estimator.write.overwrite().save(savePath)
    val loaded = LLMUncertaintyEstimator.load(savePath)

    assert(loaded.getEnsembleWeights.map(_.toSeq).contains(Seq(0.25, 0.75)))
  }

  behavior of "LLMUncertaintyEstimator optional getters"

  it should "report unset threshold and ensembleWeights as None instead of throwing" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
    assert(estimator.getThreshold.isEmpty)
    assert(estimator.getEnsembleWeights.isEmpty)
  }

  behavior of "LLMUncertaintyEstimator param validation"

  // These go through `set(param, value)` rather than the typed setters on purpose: that is
  // exactly what pyspark's _transfer_params_to_java does, so the Scala setters' own `require`s
  // never run for a Python-configured pipeline. Validation has to survive that path.

  /** Runs the same driver-side check `beforeAnnotate` performs at the start of every transform,
    * returning the failure message.
    */
  private def failureFrom(configure: LLMUncertaintyEstimator => Unit): String = {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions")
      .setOutputCol("uncertainty")
    configure(estimator)
    intercept[IllegalArgumentException](estimator.validateParams()).getMessage
  }

  it should "reject an unknown method set through the raw param" taggedAs FastTest in {
    val message = failureFrom(e => e.set(e.methods, Array("semanticEntropy", "notAMethod")))
    assert(message.contains("notAMethod"), s"unhelpful message: $message")
    assert(message.contains("Supported"), s"message should list the supported methods: $message")
  }

  it should "reject an unknown similarityBackend set through the raw param" taggedAs FastTest in {
    assert(failureFrom(e => e.set(e.similarityBackend, "cosine")).contains("cosine"))
  }

  it should "reject an empty methods array" taggedAs FastTest in {
    assert(failureFrom(e => e.set(e.methods, Array.empty[String])).contains("must not be empty"))
  }

  it should "reject repeated methods, which would make positional weights ambiguous" taggedAs FastTest in {
    val message =
      failureFrom(e => e.set(e.methods, Array("meanLogProb", "perplexity", "meanLogProb")))
    assert(message.contains("must not repeat"))
  }

  it should "reject ensembleWeights whose length does not match methods" taggedAs FastTest in {
    // Previously this indexed past the end of the weights array on an executor, surfacing as a
    // bare ArrayIndexOutOfBoundsException mid-job.
    val message = failureFrom { e =>
      e.set(e.methods, Array("meanLogProb", "perplexity", "predictiveEntropy"))
      e.set(e.ensemble, true)
      e.set(e.ensembleWeights, Array(1.0, 2.0))
    }
    assert(message.contains("ensembleWeights"))
    assert(message.contains("2") && message.contains("3"), s"should name both counts: $message")
  }

  it should "reject ensembleWeights that sum to zero, which would divide by zero" taggedAs FastTest in {
    val message = failureFrom { e =>
      e.set(e.methods, Array("meanLogProb", "perplexity"))
      e.set(e.ensemble, true)
      e.set(e.ensembleWeights, Array(0.0, 0.0))
    }
    assert(message.contains("sum to zero"))
  }

  it should "reject negative ensembleWeights" taggedAs FastTest in {
    val message = failureFrom { e =>
      e.set(e.methods, Array("meanLogProb", "perplexity"))
      e.set(e.ensemble, true)
      e.set(e.ensembleWeights, Array(-1.0, 2.0))
    }
    assert(message.contains("non-negative"))
  }

  it should "accept a valid ensembleWeights configuration" taggedAs FastTest in {
    val estimator = new LLMUncertaintyEstimator()
      .setInputCols("completions")
      .setOutputCol("uncertainty")
      .setMethods(Array("meanLogProb", "perplexity"))
      .setEnsemble(true)
      .setEnsembleWeights(Array(1.0, 3.0))
    estimator.validateParams()

    val probabilities = """[{"logprob": -0.5, "bytes": [80]}]"""
    val input = Seq(completion("Paris", 0, Map("completion_probabilities" -> probabilities)))
    val metadata = estimator.annotate(input).head.metadata
    assert(metadata.contains("uncertainty_score"))
    assert(!metadata("uncertainty_score").toDouble.isNaN)
  }

  it should "weight methods positionally when ensembling" taggedAs FastTest in {
    // meanLogProb is confidence-oriented, so its normalized contribution is -(-0.5) = 0.5;
    // perplexity is exp(0.5) - 1 = 0.6487. With weights 3:1 the average leans towards the first.
    val probabilities = """[{"logprob": -0.5, "bytes": [80]}]"""
    val input = Seq(completion("Paris", 0, Map("completion_probabilities" -> probabilities)))

    def scoreWith(weights: Array[Double]): Double =
      new LLMUncertaintyEstimator()
        .setInputCols("completions")
        .setOutputCol("uncertainty")
        .setMethods(Array("meanLogProb", "perplexity"))
        .setEnsemble(true)
        .setEnsembleWeights(weights)
        .annotate(input)
        .head
        .metadata("uncertainty_score")
        .toDouble

    val leaningMeanLogProb = scoreWith(Array(3.0, 1.0))
    val leaningPerplexity = scoreWith(Array(1.0, 3.0))
    assert(leaningPerplexity > leaningMeanLogProb)
  }
}
