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

import com.johnsnowlabs.nlp.LightPipeline
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.Row
import org.scalatest.flatspec.AnyFlatSpec

/** Exhaustive extractive-method validation (model: all_mpnet_base_v2). Runs independently of the
  * other method specs so the three lanes can execute in parallel.
  */
class SummarizationExtractiveE2ESpec extends AnyFlatSpec with SummarizationE2EBase {

  "Summarization extractive (e2e)" should "handle real documents of every size with correct metadata" taggedAs SlowTest in {
    val data = df(shortDoc, swiftDoc, wikiDoc, darwinDoc)
    val fitted = fit(summarizer("extractive").setMaxSummaryLength(80), data)

    val anns = annotations(fitted.transform(data))
    assert(anns.length == 4)
    anns.foreach { ann =>
      assert(ann.result.nonEmpty, "summary must not be empty")
      assertCoreMetadata(ann, "extractive")
      assert(ann.metadata("sentencesSelected").toInt >= 1)
      assert(
        ann.metadata("sentencesSelected").toInt <= ann.metadata("sentencesTotal").toInt,
        "selected must be <= total")
      assert(words(ann.result) <= 80 + 60, s"budget exceeded: ${words(ann.result)} words")
    }

    // extractive summaries must be verbatim subsets of the source
    val darwinSummary = anns.last.result
    ExtractiveSummarizationRanker.splitSentences(darwinSummary).foreach { s =>
      val needle = s.take(60).trim.replaceAll("\\s+", " ")
      assert(
        darwinDoc.replaceAll("\\s+", " ").contains(needle),
        s"extractive sentence not found verbatim in source: '$needle...'")
    }

    val model = summarizationStage(fitted)

    model.setMmrLambda(0.0f)
    assert(annotations(fitted.transform(df(wikiDoc))).head.result.nonEmpty)
    model.setMmrLambda(1.0f)
    assert(annotations(fitted.transform(df(wikiDoc))).head.result.nonEmpty)
    model.setMmrLambda(0.7f)

    model.setPositionBias(5.0f)
    val leadBiased = annotations(fitted.transform(df(wikiDoc))).head.result
    val firstSentence = ExtractiveSummarizationRanker.splitSentences(wikiDoc).head
    assert(
      leadBiased.replaceAll("\\s+", " ").contains(firstSentence.take(50).replaceAll("\\s+", " ")),
      "strong positionBias must select the lead sentence")
    model.setPositionBias(0.3f)

    model.setMaxSummaryLength(30)
    val tiny = annotations(fitted.transform(df(wikiDoc))).head
    model.setMaxSummaryLength(200)
    val large = annotations(fitted.transform(df(wikiDoc))).head
    assert(
      words(tiny.result) < words(large.result),
      "smaller budget must produce shorter summary")
  }

  it should "handle empty and whitespace documents without failing" taggedAs SlowTest in {
    val data = df("", "   ", shortDoc)
    val fitted = fit(summarizer("extractive"), data)
    val anns = annotations(fitted.transform(data))
    assert(anns.length == 3)
    assert(anns.last.result.nonEmpty, "non-empty doc must still be summarized")
  }

  it should "round-trip through save/load with identical output" taggedAs SlowTest in {
    val data = df(swiftDoc)
    val fitted = fit(summarizer("extractive").setMaxSummaryLength(60), data)
    val before = annotations(fitted.transform(data)).head.result

    val path = s"file:///tmp/summarization_e2e_extractive_${System.currentTimeMillis()}"
    fitted.write.overwrite().save(path)
    val after = annotations(PipelineModel.load(path).transform(data)).head.result
    assert(before == after, "extractive summarization must be deterministic across save/load")
  }

  it should "accept an explicit model override" taggedAs SlowTest in {
    val data = df(shortDoc)
    val fitted = fit(summarizer("extractive").setModel("all_mpnet_base_v2"), data)
    assert(annotations(fitted.transform(data)).head.metadata("model") == "all_mpnet_base_v2")
  }

  // Gap #1 — a single row can hold several DOCUMENT annotations (e.g. from SentenceDetector);
  // each must get its own summary. Exercises the per-document mapping in the extractive UDF.
  it should "emit one summary per DOCUMENT annotation when a row holds several" taggedAs SlowTest in {
    val doc = documentAssembler
    val sentences = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentences")
      .setExplodeSentences(false)
    val summ = summarizer("extractive").setInputCols("sentences").setMaxSummaryLength(20)
    val data = df(shortDoc)
    val fitted = new Pipeline().setStages(Array(doc, sentences, summ)).fit(data)
    val anns = annotations(fitted.transform(data))
    assert(anns.length >= 2, s"expected one summary per sentence-document, got ${anns.length}")
    anns.foreach(a => assert(a.metadata("method") == "extractive"))
  }

  // Gap #7 — offsets must be consistent for multibyte/emoji text.
  it should "produce valid offsets for unicode and emoji text" taggedAs SlowTest in {
    val uni =
      ("日本語の要約テスト。これはユニコード対応の確認です。Café déjà vu, naive facade. 🎉🚀 " * 4).trim
    val fitted = fit(summarizer("extractive").setMaxSummaryLength(40), df(uni))
    val ann = annotations(fitted.transform(df(uni))).head
    assert(ann.result.nonEmpty)
    assert(ann.begin == 0)
    assert(ann.end == math.max(0, ann.result.length - 1), "end offset must match result length")
  }

  // Gap #3/#4 — model params must survive save/load, and the delegate must deserialize from disk
  // (close the in-memory copy first so the loaded model cannot reuse it).
  it should "preserve params and reload the delegate from disk" taggedAs SlowTest in {
    val summ =
      summarizer("extractive").setMaxSummaryLength(44).setMmrLambda(0.6f).setPositionBias(0.8f)
    val fitted = fit(summ, df(shortDoc))
    val path = s"file:///tmp/summ_model_params_${System.currentTimeMillis()}"
    fitted.write.overwrite().save(path)
    summarizationStage(fitted).close() // extractive: no-op, but proves independence

    val loaded = PipelineModel.load(path)
    val lm = loaded.stages.collectFirst { case m: SummarizationModel => m }.get
    assert(lm.getMaxSummaryLength == 44)
    assert(lm.getMmrLambda == 0.6f)
    assert(lm.getPositionBias == 0.8f)
    assert(lm.getMethod == "extractive")
    assert(annotations(loaded.transform(df(shortDoc))).head.result.nonEmpty)
  }

  // Gap #10 — several DOCUMENT annotations in one row that share character offsets (all begin at
  // 0) must not bleed sentences into each other's summary. Guards the per-document isolation of
  // the extractive path.
  it should "not mix sentences across DOCUMENT annotations that share offsets" taggedAs SlowTest in {
    val climate =
      "Average surface temperatures keep climbing across every measured region on the planet. " +
        "Coastal areas report shifting precipitation patterns and far more frequent flooding. " +
        "Agricultural yields are projected to fall sharply in several of the affected regions. " +
        "Governments have started funding adaptation and coastal resilience programmes."
    val finance =
      "The central bank held its benchmark interest rate unchanged at the latest meeting. " +
        "Analysts expect consumer inflation to ease gradually over the coming few quarters."

    val model =
      summarizationStage(fit(summarizer("extractive").setMaxSummaryLength(40), df(shortDoc)))
    val data = documentsDF(Seq(climate, finance)) // one row, two overlapping-offset documents
    val anns = annotations(model.transform(data))
    assert(anns.length == 2, s"expected one summary per document, got ${anns.length}")

    val norm = (s: String) => s.replaceAll("\\s+", " ").trim
    val climateNorm = norm(climate)
    val financeNorm = norm(finance)

    // annotations come back ordered by annotation index: 0 = climate, 1 = finance
    ExtractiveSummarizationRanker.splitSentences(anns.head.result).foreach { s =>
      assert(
        climateNorm.contains(norm(s)),
        s"first document's summary must contain only its own sentences: '${norm(s)}'")
    }
    ExtractiveSummarizationRanker.splitSentences(anns(1).result).foreach { s =>
      assert(
        financeNorm.contains(norm(s)),
        s"second document's summary must contain only its own sentences: '${norm(s)}'")
    }
  }

  // Gap #11 — a genuinely empty input annotation array must yield an empty output array (empty in
  // -> empty out), while a real document in the same batch is still summarized.
  it should "return an empty summary array for an empty input annotation array" taggedAs SlowTest in {
    val model =
      summarizationStage(fit(summarizer("extractive").setMaxSummaryLength(30), df(shortDoc)))
    val data = documentsDF(Seq.empty[String], Seq(shortDoc)) // row 0 empty, row 1 one document
    val sizes = model
      .transform(data)
      .select("summary")
      .collect()
      .map(r => Option(r.getSeq[Row](0)).getOrElse(Seq.empty).length)
    assert(sizes.toSeq == Seq(0, 1), s"expected [0, 1] summaries per row, got ${sizes.toSeq}")
  }

  // Gap #5 — DataFrame-only contract: LightPipeline must not fabricate a summary.
  it should "not fabricate a summary under LightPipeline" taggedAs SlowTest in {
    val fitted = fit(summarizer("extractive").setMaxSummaryLength(20), df(shortDoc))
    val light = new LightPipeline(fitted)
    scala.util.Try(light.annotate(shortDoc)) match {
      case scala.util.Success(m) =>
        val s = m.getOrElse("summary", Seq.empty)
        assert(
          s.isEmpty || s.forall(_.isEmpty),
          "LightPipeline is unsupported and must not return a real summary")
      case scala.util.Failure(_) => succeed
    }
  }
}
