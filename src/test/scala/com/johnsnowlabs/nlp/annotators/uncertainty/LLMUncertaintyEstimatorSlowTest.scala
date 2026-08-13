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

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFModel
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.MPNetEmbeddings
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

/** End-to-end pipeline tests over real GGUF models. Requires `models/Qwen3-1.7B-Q4_K_M.gguf`
  * locally, per the same convention as `AutoGGUFModelTest`. Absolute score values are not stable
  * enough to assert on; these tests assert the *ordering* between a well-known-answer prompt and
  * a deliberately ambiguous one, which is the real end-to-end signal.
  */
class LLMUncertaintyEstimatorSlowTest extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  behavior of "LLMUncertaintyEstimator end-to-end (black box, embeddings backend)"

  it should "score a well-known-answer prompt as more confident than an ambiguous one" taggedAs SlowTest in {
    val documentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("document")

    val llm = AutoGGUFModel
      .loadSavedModel("models/Qwen3-1.7B-Q4_K_M.gguf", ResourceHelper.spark)
      .setInputCols("document")
      .setOutputCol("completions")
      .setNumSamples(5)
      .setTemperature(0.8f)
      .setNPredict(120)
      .setBatchSize(5)
      // Qwen3 reasons by default (<think>...</think> before the real answer). Without
      // stripping it, semantic-entropy clustering embeds the reasoning preamble - which is
      // near-identical boilerplate across samples regardless of how different the actual
      // answers are - and collapses everything into one cluster, inverting the score.
      .setRemoveThinkingTag("think")

    val embeddings = MPNetEmbeddings
      .pretrained()
      .setInputCols("completions")
      .setOutputCol("sample_embeddings")

    val uncertainty = new LLMUncertaintyEstimator()
      .setInputCols("completions", "sample_embeddings")
      .setOutputCol("uncertainty")
      .setMethods(Array("semanticEntropy"))

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, llm, embeddings, uncertainty))

    val data =
      Seq("What is the capital of France? Answer in one word.", "Write a single random word.")
        .toDF("text")

    val result = pipeline.fit(data).transform(data)
    val rows = Annotation.collect(result, "uncertainty")
    assert(rows.length == 2)

    val wellKnownUncertainty = rows(0).head.metadata("uncertainty_score").toDouble
    val ambiguousUncertainty = rows(1).head.metadata("uncertainty_score").toDouble

    println(s"Well-known-answer uncertainty: $wellKnownUncertainty")
    println(s"Ambiguous-prompt uncertainty: $ambiguousUncertainty")
    assert(
      ambiguousUncertainty >= wellKnownUncertainty,
      "A deliberately ambiguous prompt should not be scored as more confident than a " +
        "well-known-answer prompt")
  }

  behavior of "LLMUncertaintyEstimator end-to-end (white box)"

  it should "compute mars uncertainty using AutoGGUFModel logprobs and MarsTokenImportance" taggedAs SlowTest in {
    val documentAssembler =
      new DocumentAssembler().setInputCol("question").setOutputCol("question_doc")
    val answerAssembler =
      new DocumentAssembler().setInputCol("question").setOutputCol("answer_source")

    val llm = AutoGGUFModel
      .loadSavedModel("models/Qwen3-1.7B-Q4_K_M.gguf", ResourceHelper.spark)
      .setInputCols("answer_source")
      .setOutputCol("completions")
      .setOutputLogProbs(true)
      .setNProbs(5)
      .setNPredict(15)
      .setTemperature(0.1f)

    val mars = MarsTokenImportance
      .load("models/mars_token_importance")
      .setInputCols("question_doc", "completions")
      .setOutputCol("scored_completions")

    val uncertainty = new LLMUncertaintyEstimator()
      .setInputCols("scored_completions")
      .setOutputCol("uncertainty")
      .setMethods(Array("mars", "meanLogProb", "perplexity"))

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, answerAssembler, llm, mars, uncertainty))

    val data = Seq("What is the capital of France? Answer in one word.").toDF("question")
    val result = pipeline.fit(data).transform(data)

    val annotation = Annotation.collect(result, "uncertainty").head.head
    assert(annotation.metadata.contains("mars"))
    assert(annotation.metadata.contains("mean_log_prob"))
    assert(annotation.metadata.contains("perplexity"))
    assert(annotation.metadata("perplexity").toDouble >= 1.0)
  }
}
