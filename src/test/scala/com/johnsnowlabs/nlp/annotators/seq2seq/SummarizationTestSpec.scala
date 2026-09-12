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

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Row}
import org.scalatest.flatspec.AnyFlatSpec

class SummarizationTestSpec extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  private val shortText =
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
      "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand " +
      "customers were scheduled to be affected by the shutoffs which were expected to last " +
      "through at least midday tomorrow."

  private val longText = {
    val paragraph =
      "The research team analyzed climate data from over two hundred weather stations. " +
        "Temperatures rose steadily across all measured regions during the last decade. " +
        "Precipitation patterns shifted significantly in coastal areas. " +
        "The models predict further changes if emissions continue at current rates. " +
        "Agricultural yields are expected to decline in affected regions. " +
        "Local governments have begun implementing adaptation strategies. "
    paragraph * 40 // ~15k chars, forces chunking for encoder_decoder budgets
  }

  private def collectAnnotations(df: DataFrame, colName: String): Seq[Annotation] =
    df.select(colName)
      .collect()
      .flatMap(row => Option(row.getSeq[Row](0)).getOrElse(Seq.empty).map(Annotation(_)))
      .toSeq

  private def pipelineFor(summarizer: Summarization): Pipeline = {
    val documentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("document")
    new Pipeline().setStages(Array(documentAssembler, summarizer))
  }

  // ------------------------------------------------------------------------------------------
  // FastTest: pure logic
  // ------------------------------------------------------------------------------------------

  "ExtractiveSummarizationRanker" should "split and pack sentences within budget" taggedAs FastTest in {
    val sentences = ExtractiveSummarizationRanker.splitSentences(
      "First sentence here. Second sentence follows. Third one ends it.")
    assert(sentences.length == 3)

    val chunks = ExtractiveSummarizationRanker.packChunks(sentences, budgetChars = 45, 1)
    assert(chunks.length > 1)
    assert(chunks.forall(_.length <= 45 + 25)) // overlap seeding tolerance
  }

  it should "produce single chunk when text fits" taggedAs FastTest in {
    val sentences = Array("Short one.", "Another short.")
    val chunks = ExtractiveSummarizationRanker.packChunks(sentences, budgetChars = 1000, 1)
    assert(chunks.length == 1)
  }

  it should "carry overlap sentences between consecutive chunks" taggedAs FastTest in {
    val sentences = Array("Alpha one.", "Bravo two.", "Charlie three.", "Delta four.")
    val chunks = ExtractiveSummarizationRanker.packChunks(sentences, budgetChars = 22, 1)
    assert(chunks.length >= 2)
    // the second chunk is seeded with the last sentence of the first (overlap continuity)
    assert(
      chunks(1).startsWith("Bravo two."),
      s"expected overlap 'Bravo two.' to seed the second chunk, got '${chunks(1)}'")
  }

  it should "not emit a trailing chunk that only repeats the previous overlap" taggedAs FastTest in {
    val sentences = (1 to 12).map(i => s"Sentence number $i ends here.").toArray
    for (budget <- Seq(20, 30, 45, 60); overlap <- Seq(1, 2)) {
      val chunks = ExtractiveSummarizationRanker.packChunks(sentences, budget, overlap)
      chunks.sliding(2).foreach {
        case Array(prev, next) =>
          assert(
            !prev.endsWith(next),
            s"trailing all-overlap chunk not dropped (budget=$budget, overlap=$overlap)")
        case _ => // single-chunk window, nothing to compare
      }
    }
  }

  it should "unwrap a JSON-string-literal shaped llm response" taggedAs FastTest in {
    val wrapped = "\"- First point.  \\n- Second point.  \\n- Third point.\""
    val unwrapped = SummarizationModel.unwrapJsonStringLiteral(wrapped)
    assert(unwrapped == "- First point.  \n- Second point.  \n- Third point.")
    assert(!unwrapped.contains("\\n"), "literal backslash-n must become a real newline")
    assert(!unwrapped.startsWith("\""), "wrapping quote must be stripped")
  }

  it should "leave plain (non-wrapped) llm responses unchanged" taggedAs FastTest in {
    val plain = "- First point.\n- Second point."
    assert(SummarizationModel.unwrapJsonStringLiteral(plain) == plain)
  }

  it should "keep quotes on a summary that merely starts and ends with a quotation mark" taggedAs FastTest in {
    // quoted speech, not a JSON-string literal: no escape sequences inside the quotes
    val quoted = "\"We will reduce the risk of wildfires,\" the utility said."
    assert(SummarizationModel.unwrapJsonStringLiteral(quoted) == quoted)
    val fullyQuoted = "\"The utility scheduled blackouts to reduce wildfire risk.\""
    assert(SummarizationModel.unwrapJsonStringLiteral(fullyQuoted) == fullyQuoted)
  }

  it should "not turn an escaped backslash followed by n into a newline" taggedAs FastTest in {
    // the JSON escape \\n denotes a literal backslash followed by the letter n
    val wrapped = "\"path C:\\\\new stays literal.\\nSecond line.\""
    val unwrapped = SummarizationModel.unwrapJsonStringLiteral(wrapped)
    assert(unwrapped == "path C:\\new stays literal.\nSecond line.")
  }

  it should "rank lead and central sentences higher" taggedAs FastTest in {
    // sentence 0 and 1 similar (recurring topic), sentence 2 orthogonal
    val embs = Array(Array(1.0f, 0.1f, 0.0f), Array(0.9f, 0.2f, 0.0f), Array(0.0f, 0.0f, 1.0f))
    val scores = ExtractiveSummarizationRanker.centralityScores(embs)
    assert(scores(0) > scores(2), "lead + central sentence must outrank isolated late sentence")
  }

  it should "respect the word budget and avoid redundant picks in MMR" taggedAs FastTest in {
    val embs = Array(
      Array(1.0f, 0.0f),
      Array(1.0f, 0.0f), // duplicate of 0
      Array(0.0f, 1.0f))
    val scores = Array(1.0, 0.99, 0.5)
    val wordCounts = Array(5, 5, 5)
    // with a redundancy-leaning lambda the duplicate must lose to the diverse sentence
    val selected =
      ExtractiveSummarizationRanker.mmrSelect(embs, scores, wordCounts, budgetWords = 10, 0.5)
    assert(selected.length == 2)
    assert(selected.contains(0))
    assert(selected.contains(2), "MMR should prefer the diverse sentence over the duplicate")
    // with a purely relevance-driven lambda the duplicate wins instead
    val relevanceOnly =
      ExtractiveSummarizationRanker.mmrSelect(embs, scores, wordCounts, budgetWords = 10, 1.0)
    assert(relevanceOnly.sameElements(Array(0, 1)))
  }

  it should "always select at least one sentence" taggedAs FastTest in {
    val embs = Array(Array(1.0f, 0.0f))
    val selected =
      ExtractiveSummarizationRanker.mmrSelect(embs, Array(1.0), Array(50), budgetWords = 10, 0.7)
    assert(selected.length == 1)
  }

  "Summarization" should "validate parameter values" taggedAs FastTest in {
    val summarizer = new Summarization()
    assertThrows[IllegalArgumentException](summarizer.setMethod("magic"))
    assertThrows[IllegalArgumentException](summarizer.setSummaryStyle("epic"))
    assertThrows[IllegalArgumentException](summarizer.setLongDocumentStrategy("never"))
    assertThrows[IllegalArgumentException](summarizer.setMaxSummaryLength(0))
    assertThrows[IllegalArgumentException](summarizer.setMmrLambda(1.5f))
    summarizer.setMethod("extractive").setSummaryStyle("bullets")
    assert(summarizer.getMethod == "extractive")
  }

  // ------------------------------------------------------------------------------------------
  // SlowTest: end-to-end per method (downloads pretrained models)
  // ------------------------------------------------------------------------------------------

  "Summarization extractive" should "summarize and report metadata" taggedAs SlowTest in {
    val data = Seq(shortText, longText).toDF("text")

    val summarizer = new Summarization()
      .setInputCols("document")
      .setOutputCol("summary")
      .setMethod("extractive")
      .setMaxSummaryLength(60)

    val model = pipelineFor(summarizer).fit(data)
    val result = model.transform(data)
    val annotations = collectAnnotations(result, "summary")

    assert(annotations.length == 2)
    annotations.foreach { ann =>
      assert(ann.result.nonEmpty, "summary must not be empty")
      assert(ann.metadata("method") == "extractive")
      assert(ann.metadata("model").nonEmpty)
      assert(ann.metadata("sentencesSelected").toInt >= 1)
      // extractive summaries only contain source sentences
      assert(ExtractiveSummarizationRanker.countWords(ann.result) <= 60 + 40)
    }
  }

  it should "round-trip through PipelineModel save/load" taggedAs SlowTest in {
    val data = Seq(shortText).toDF("text")
    val summarizer = new Summarization()
      .setInputCols("document")
      .setOutputCol("summary")
      .setMethod("extractive")
      .setMaxSummaryLength(40)

    val fitted = pipelineFor(summarizer).fit(data)
    val path = s"file:///tmp/summarization_extractive_pipeline_${System.currentTimeMillis()}"
    fitted.write.overwrite().save(path)

    val loaded = PipelineModel.load(path)
    val annotations = collectAnnotations(loaded.transform(data), "summary")
    assert(annotations.head.result.nonEmpty)
    assert(annotations.head.metadata("method") == "extractive")
  }

  "Summarization encoder_decoder" should "summarize short and long documents" taggedAs SlowTest in {
    val data = Seq(shortText, longText).toDF("text")

    val summarizer = new Summarization()
      .setInputCols("document")
      .setOutputCol("summary")
      .setMethod("encoder_decoder")
      .setMaxSummaryLength(60)

    val model = pipelineFor(summarizer).fit(data)
    val result = model.transform(data)
    val annotations = collectAnnotations(result, "summary")

    assert(annotations.length == 2)
    annotations.foreach { ann =>
      assert(ann.result.nonEmpty, "summary must not be empty")
      assert(ann.metadata("method") == "encoder_decoder")
    }
    // the long document must have used more than one chunk
    val longMeta = annotations.map(_.metadata).maxBy(_("originalTokensEst").toInt)
    assert(longMeta("numChunks").toInt > 1, "long document should be chunked")
  }

  it should "round-trip through PipelineModel save/load" taggedAs SlowTest in {
    val data = Seq(shortText).toDF("text")
    val summarizer = new Summarization()
      .setInputCols("document")
      .setOutputCol("summary")
      .setMethod("encoder_decoder")
      .setMaxSummaryLength(40)

    val fitted = pipelineFor(summarizer).fit(data)
    val path = s"file:///tmp/summarization_encdec_pipeline_${System.currentTimeMillis()}"
    fitted.write.overwrite().save(path)

    val loaded = PipelineModel.load(path)
    val annotations = collectAnnotations(loaded.transform(data), "summary")
    assert(annotations.head.result.nonEmpty)
  }

  "Summarization llm" should "summarize with the default GGUF model" taggedAs SlowTest in {
    val data = Seq(shortText).toDF("text")

    val summarizer = new Summarization()
      .setInputCols("document")
      .setOutputCol("summary")
      .setMethod("llm")
      .setMaxSummaryLength(80)
      .setSummaryStyle("concise")

    val model = pipelineFor(summarizer).fit(data)
    val result = model.transform(data)
    val annotations = collectAnnotations(result, "summary")

    assert(annotations.nonEmpty)
    val ann = annotations.head
    assert(ann.result.nonEmpty, "summary must not be empty")
    assert(!ann.result.contains("<think>"), "reasoning tags must be stripped")
    assert(ann.metadata("method") == "llm")
    assert(ann.metadata("engine") == "llamacpp")
  }

  // ------------------------------------------------------------------------------------------
  // FastTest: serialization + schema contract (no model download)
  // ------------------------------------------------------------------------------------------

  "Summarization estimator" should "round-trip every parameter through save/load" taggedAs FastTest in {
    ResourceHelper.spark // ensure an active local SparkSession for the ML writer
    val s = new Summarization()
      .setInputCols("document")
      .setOutputCol("summary")
      .setMethod("extractive")
      .setModel("custom_model")
      .setMaxSummaryLength(123)
      .setMinSummaryLength(7)
      .setSummaryStyle("bullets")
      .setFocus("main findings")
      .setLongDocumentStrategy("truncate")
      .setNumBeams(6)
      .setTemperature(0.4f)
      .setTopP(0.85f)
      .setChunkSize(999)
      .setChunkOverlap(3)
      .setMmrLambda(0.55f)
      .setPositionBias(0.9f)
      .setGpuLayers(0)
    val path = s"file:///tmp/summ_estimator_params_${System.currentTimeMillis()}"
    s.write.overwrite().save(path)
    val loaded = Summarization.load(path)
    assert(loaded.getMethod == "extractive")
    assert(loaded.getModel == "custom_model")
    assert(loaded.getMaxSummaryLength == 123)
    assert(loaded.getMinSummaryLength == 7)
    assert(loaded.getSummaryStyle == "bullets")
    assert(loaded.getFocus == "main findings")
    assert(loaded.getLongDocumentStrategy == "truncate")
    assert(loaded.getNumBeams == 6)
    assert(loaded.getTemperature == 0.4f)
    assert(loaded.getTopP == 0.85f)
    assert(loaded.getChunkSize == 999)
    assert(loaded.getChunkOverlap == 3)
    assert(loaded.getMmrLambda == 0.55f)
    assert(loaded.getPositionBias == 0.9f)
    assert(loaded.getGpuLayers == 0)
  }

  it should "fail fast at fit when minSummaryLength exceeds maxSummaryLength" taggedAs FastTest in {
    val documentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val data = documentAssembler.transform(Seq("some text").toDF("text"))
    val summarizer = new Summarization()
      .setInputCols("document")
      .setOutputCol("summary")
      .setMethod("extractive")
      .setMinSummaryLength(300)
      .setMaxSummaryLength(100)
    // the require fires before any model resolution/download
    assertThrows[IllegalArgumentException](summarizer.fit(data))
  }

  it should "reject inputs that are not DOCUMENT annotations" taggedAs FastTest in {
    val documentAssembler =
      new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token")
    val base = documentAssembler.transform(Seq("hello world foo bar").toDF("text"))
    val tokenized = tokenizer.fit(base).transform(base)

    val model = new SummarizationModel()
      .setInputCols("token")
      .setOutputCol("summary")
      .setMethod("extractive")
    // validate() runs before any delegate is touched, so this needs no model
    assertThrows[IllegalArgumentException](model.transform(tokenized))
  }
}
