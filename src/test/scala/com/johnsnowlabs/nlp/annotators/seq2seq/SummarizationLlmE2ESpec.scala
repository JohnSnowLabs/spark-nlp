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
package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.PipelineModel
import org.scalatest.flatspec.AnyFlatSpec

/** Exhaustive LLM-method validation (model: qwen3_4b_q8_0_gguf, CPU via gpuLayers=0). Runs
  * independently of the other method specs so the three lanes can execute in parallel.
  */
class SummarizationLlmE2ESpec extends AnyFlatSpec with SummarizationE2EBase {

  private def llm: Summarization = summarizer("llm").setGpuLayers(0)

  "Summarization llm (e2e)" should "summarize with every prompt-shaping parameter" taggedAs SlowTest in {
    val data = df(shortDoc)
    val fitted = fit(llm.setMaxSummaryLength(80), data)

    val concise = annotations(fitted.transform(data)).head
    assert(concise.result.nonEmpty, "summary must not be empty")
    assertCoreMetadata(concise, "llm")
    assert(concise.metadata("engine") == "llamacpp")
    assert(!concise.result.contains("<think>"), "think tags must be stripped")
    assert(words(concise.result) <= 80 * 3, s"summary too long: ${words(concise.result)} words")

    val model = summarizationStage(fitted)

    model.setSummaryStyle("bullets")
    assert(annotations(fitted.transform(data)).head.result.nonEmpty)
    model.setSummaryStyle("detailed")
    assert(annotations(fitted.transform(data)).head.result.nonEmpty)
    model.setSummaryStyle("concise")

    model.setFocus("number of affected customers")
    assert(annotations(fitted.transform(data)).head.result.nonEmpty)
    model.setFocus("")

    model.setTemperature(0.7f)
    model.setTopP(0.95f)
    assert(annotations(fitted.transform(data)).head.result.nonEmpty)
    model.setTemperature(0.2f)
    model.setTopP(0.9f)

    model.setMaxSummaryLength(40)
    assert(annotations(fitted.transform(data)).head.result.nonEmpty)
    model.setMaxSummaryLength(80)
  }

  it should "handle a real long document hierarchically and via truncate" taggedAs SlowTest in {
    val data = df(swiftDoc)
    val fitted = fit(llm.setMaxSummaryLength(100).setChunkSize(1200), data)

    val hier = annotations(fitted.transform(data)).head
    assert(hier.result.nonEmpty, "hierarchical summary must not be empty")
    assert(hier.metadata("numChunks").toInt > 1, "document must have been chunked")
    assert(!hier.result.contains("<think>"))

    val model = summarizationStage(fitted)
    model.setLongDocumentStrategy("truncate")
    val trunc = annotations(fitted.transform(data)).head
    assert(trunc.result.nonEmpty)
    assert(trunc.metadata("numChunks").toInt == 1, "truncate must not chunk")
    model.setLongDocumentStrategy("auto")
  }

  it should "round-trip through save/load and still transform" taggedAs SlowTest in {
    val data = df(shortDoc)
    val fitted = fit(llm.setMaxSummaryLength(50), data)
    val path = s"file:///tmp/summarization_e2e_llm_${System.currentTimeMillis()}"
    fitted.write.overwrite().save(path)
    val ann = annotations(PipelineModel.load(path).transform(data)).head
    assert(ann.result.nonEmpty)
    assert(ann.metadata("method") == "llm")
  }

  // Gap #12 — task params are mutable on the fitted model, so minSummaryLength can be pushed
  // above maxSummaryLength after fit (the estimator only rejects it at fit time). The prompt
  // builder must clamp instead of emitting an incoherent "between 400 and 60 words" instruction.
  it should "tolerate minSummaryLength greater than maxSummaryLength on the fitted model" taggedAs SlowTest in {
    val fitted = fit(llm.setMaxSummaryLength(60), df(shortDoc))
    val model = summarizationStage(fitted)
    model.setMinSummaryLength(400) // exceeds max; must be clamped in the prompt, not crash
    val ann = annotations(fitted.transform(df(shortDoc))).head
    assert(ann.result.nonEmpty, "summary must still be produced when min > max")
    model.setMinSummaryLength(20)
  }
}
