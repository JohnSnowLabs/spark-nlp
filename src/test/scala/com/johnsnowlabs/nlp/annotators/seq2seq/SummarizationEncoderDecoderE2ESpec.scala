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

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.flatspec.AnyFlatSpec

/** Exhaustive encoder-decoder validation (model: distilbart_xsum_12_6). Runs independently of the
  * other method specs so the three lanes can execute in parallel.
  */
class SummarizationEncoderDecoderE2ESpec extends AnyFlatSpec with SummarizationE2EBase {

  "Summarization encoder_decoder (e2e)" should "summarize real documents, chunking long ones" taggedAs SlowTest in {
    val data = df(shortDoc, swiftDoc, wikiDoc)
    val fitted = fit(summarizer("encoder_decoder").setMaxSummaryLength(80), data)

    val anns = annotations(fitted.transform(data))
    assert(anns.length == 3)
    anns.foreach { ann =>
      assert(ann.result.nonEmpty, "summary must not be empty")
      assertCoreMetadata(ann, "encoder_decoder")
    }
    val byLen = anns.sortBy(_.metadata("originalTokensEst").toInt)
    assert(byLen.head.metadata("numChunks").toInt == 1, "short doc must not be chunked")
    assert(byLen(1).metadata("numChunks").toInt > 1, "swift doc must be chunked")
    assert(byLen(2).metadata("numChunks").toInt > 1, "wiki doc must be chunked")

    val model = summarizationStage(fitted)

    model.setNumBeams(1)
    assert(annotations(fitted.transform(df(shortDoc))).head.result.nonEmpty, "greedy must work")
    model.setNumBeams(4)

    model.setMaxSummaryLength(30)
    model.setMinSummaryLength(5)
    assert(annotations(fitted.transform(df(shortDoc))).head.result.nonEmpty)
    model.setMaxSummaryLength(80)
    model.setMinSummaryLength(20)

    model.setLongDocumentStrategy("truncate")
    val truncated = annotations(fitted.transform(df(wikiDoc))).head
    assert(truncated.metadata("numChunks").toInt == 1, "truncate must not chunk")
    assert(truncated.metadata("longDocumentStrategy") == "truncate")
    model.setLongDocumentStrategy("auto")

    model.setChunkSize(200)
    model.setChunkOverlap(0)
    val fineChunked = annotations(fitted.transform(df(swiftDoc))).head
    assert(
      fineChunked.metadata("numChunks").toInt > anns(1).metadata("numChunks").toInt,
      "smaller chunkSize must produce more chunks")
    model.setChunkSize(0)
    model.setChunkOverlap(1)
  }

  it should "round-trip through save/load and still transform" taggedAs SlowTest in {
    val data = df(shortDoc)
    val fitted = fit(summarizer("encoder_decoder").setMaxSummaryLength(40), data)
    val path = s"file:///tmp/summarization_e2e_encdec_${System.currentTimeMillis()}"
    fitted.write.overwrite().save(path)
    val ann = annotations(PipelineModel.load(path).transform(data)).head
    assert(ann.result.nonEmpty)
    assert(ann.metadata("method") == "encoder_decoder")
  }

  // Gap #1 — the generative path explodes/regroups per DOCUMENT annotation and sorts them back
  // by index (the struct-with-map ordering path). Verify a multi-annotation row round-trips.
  it should "align one summary per DOCUMENT annotation in a multi-annotation row" taggedAs SlowTest in {
    val doc = documentAssembler
    val sentences = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentences")
      .setExplodeSentences(false)
    val summ = summarizer("encoder_decoder").setInputCols("sentences").setMaxSummaryLength(20)
    val data = df(shortDoc)
    val fitted = new Pipeline().setStages(Array(doc, sentences, summ)).fit(data)
    val anns = annotations(fitted.transform(data))
    assert(anns.length >= 2, s"expected a summary per sentence, got ${anns.length}")
    anns.foreach(a => assert(a.metadata("method") == "encoder_decoder"))
  }

  // Gap #2 — force the hierarchical reduce to run several rounds: many small chunks whose
  // combined intermediate summaries still exceed the budget, recursing until they fit.
  it should "summarize across multiple hierarchical reduce rounds" taggedAs SlowTest in {
    val multiRoundDoc = ((shortDoc + " ") * 6).trim
    val summ = summarizer("encoder_decoder")
      .setMaxSummaryLength(40)
      .setChunkSize(100)
      .setLongDocumentStrategy("hierarchical")
    val fitted = fit(summ, df(multiRoundDoc))
    val ann = annotations(fitted.transform(df(multiRoundDoc))).head
    assert(ann.result.nonEmpty, "multi-round summary must not be empty")
    assert(ann.metadata("numChunks").toInt > 1, "document must be chunked")
    assert(ann.metadata("longDocumentStrategy") == "hierarchical")
  }

  // Gap #6 — empty / whitespace inputs must not crash and must still yield one row each.
  it should "handle empty and whitespace inputs" taggedAs SlowTest in {
    val data = df("", "   ", shortDoc)
    val fitted = fit(summarizer("encoder_decoder").setMaxSummaryLength(30), data)
    val anns = annotations(fitted.transform(data))
    assert(anns.length == 3)
    assert(anns.count(_.result.nonEmpty) >= 1, "the non-empty document must be summarized")
  }

  // Gap #8 — a smaller maxSummaryLength must actually bound the generated output.
  it should "let maxSummaryLength bound the output length" taggedAs SlowTest in {
    val fitted = fit(summarizer("encoder_decoder").setMaxSummaryLength(120), df(shortDoc))
    val model = summarizationStage(fitted)
    model.setMaxSummaryLength(120)
    val long = annotations(fitted.transform(df(shortDoc))).head.result
    model.setMaxSummaryLength(12)
    val short = annotations(fitted.transform(df(shortDoc))).head.result
    assert(words(short) <= words(long), "smaller budget must not produce a longer summary")
    assert(words(short) <= 20, "a 12-token budget must produce a short summary")
  }

  // Gap #9 — regression guard for the coalesce(1) fix: many input partitions must not
  // reintroduce the BART concurrency shape crash.
  it should "stay correct when the input is spread across many partitions" taggedAs SlowTest in {
    val data = df(swiftDoc).repartition(8)
    val fitted = fit(summarizer("encoder_decoder").setMaxSummaryLength(50), data)
    val ann = annotations(fitted.transform(data)).head
    assert(ann.result.nonEmpty)
    assert(ann.metadata("numChunks").toInt > 1)
  }

  // Gap #10 — a genuinely empty input annotation array must yield an empty output array, while a
  // real document in the same batch is still summarized (empty in -> empty out).
  it should "return an empty summary array for an empty input annotation array" taggedAs SlowTest in {
    val model =
      summarizationStage(fit(summarizer("encoder_decoder").setMaxSummaryLength(30), df(shortDoc)))
    val data = documentsDF(Seq.empty[String], Seq(shortDoc)) // row 0 empty, row 1 one document
    val sizes = model
      .transform(data)
      .select("summary")
      .collect()
      .map(r => Option(r.getSeq[org.apache.spark.sql.Row](0)).getOrElse(Seq.empty).length)
    assert(sizes.toSeq == Seq(0, 1), s"expected [0, 1] summaries per row, got ${sizes.toSeq}")
  }

  // Gap #11 — hierarchical chunking with a positive sentence overlap must run end-to-end and
  // still produce a summary (exercises packChunks overlap seeding in the generative path).
  it should "chunk long documents with sentence overlap between chunks" taggedAs SlowTest in {
    val summ = summarizer("encoder_decoder")
      .setMaxSummaryLength(40)
      .setChunkSize(150)
      .setChunkOverlap(2)
      .setLongDocumentStrategy("hierarchical")
    val fitted = fit(summ, df(wikiDoc))
    val ann = annotations(fitted.transform(df(wikiDoc))).head
    assert(ann.result.nonEmpty, "overlapped-chunk summary must not be empty")
    assert(ann.metadata("numChunks").toInt > 1, "document must be chunked")
  }
}
