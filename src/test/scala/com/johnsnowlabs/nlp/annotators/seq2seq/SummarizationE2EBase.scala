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
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Row}

/** Shared fixtures/helpers for the split per-method Summarization end-to-end specs. Each concrete
  * spec exercises exactly one method (and therefore one pretrained model), so the specs can run
  * as independent JVM processes in parallel without racing on the shared model download cache.
  *
  * The long documents are generated in-memory from distinct, self-contained paragraphs so the
  * suite carries no external corpora. This is sufficient because the long-document assertions
  * (chunk counts, hierarchical rounds, verbatim-subset selection, save/load) are content
  * agnostic: they only require documents that exceed the model context and split into several
  * sentences.
  */
trait SummarizationE2EBase {

  import ResourceHelper.spark.implicits._

  /** Repeats a paragraph until the text is at least `minChars` long. */
  private def longDoc(paragraph: String, minChars: Int): String = {
    val sb = new StringBuilder
    while (sb.length < minChars) sb.append(paragraph).append(' ')
    sb.toString.trim
  }

  private val economyParagraph =
    "The national statistics office reported that industrial output rose modestly last quarter. " +
      "Manufacturers cited stronger export demand and easing supply constraints as the main " +
      "drivers. Economists cautioned that rising energy costs could weigh on margins in the " +
      "months ahead. Consumer spending held steady while household savings edged lower. Analysts " +
      "expect the central bank to keep interest rates unchanged at its next meeting."

  private val climateParagraph =
    "Average surface temperatures have continued to climb across most measured regions. " +
      "Researchers link the trend to sustained greenhouse gas emissions from energy and " +
      "transport. Coastal areas reported shifting precipitation patterns and more frequent " +
      "flooding. Agricultural yields are projected to decline in several affected regions. Local " +
      "governments have begun funding adaptation and resilience programmes."

  private val biologyParagraph =
    "The survey documented considerable variation among the sampled populations. Individuals with " +
      "traits suited to the local environment tended to leave more descendants. Over successive " +
      "generations these advantageous characteristics became more common. Isolated groups " +
      "gradually diverged as conditions differed between habitats. The authors argue that small " +
      "inherited differences accumulate into pronounced change over long periods."

  // long, self-contained documents (no external corpora); sized to exceed the model context
  protected lazy val swiftDoc: String = longDoc(economyParagraph, 6000)
  protected lazy val wikiDoc: String = longDoc(climateParagraph, 12000)
  protected lazy val darwinDoc: String = longDoc(biologyParagraph, 24000)

  protected val shortDoc: String =
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds " +
      "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand " +
      "customers were scheduled to be affected by the shutoffs which were expected to last " +
      "through at least midday tomorrow."

  protected def df(texts: String*): DataFrame = texts.toDF("text")

  protected def documentAssembler: DocumentAssembler =
    new DocumentAssembler().setInputCol("text").setOutputCol("document")

  protected def fit(summarizer: Summarization, data: DataFrame): PipelineModel =
    new Pipeline().setStages(Array(documentAssembler, summarizer)).fit(data)

  protected def summarizationStage(fitted: PipelineModel): SummarizationModel =
    fitted.stages.collectFirst { case m: SummarizationModel => m }.get

  protected def annotations(result: DataFrame): Seq[Annotation] =
    result
      .select("summary")
      .collect()
      .flatMap(row => Option(row.getSeq[Row](0)).getOrElse(Seq.empty).map(Annotation(_)))
      .toSeq

  protected def words(s: String): Int = ExtractiveSummarizationRanker.countWords(s)

  protected def assertCoreMetadata(ann: Annotation, method: String): Unit = {
    assert(ann.metadata("method") == method, s"metadata.method must be $method")
    assert(ann.metadata("model").nonEmpty, "metadata.model must be present")
    assert(ann.metadata("engine").nonEmpty, "metadata.engine must be present")
    assert(ann.metadata.contains("originalTokensEst"), "originalTokensEst missing")
    assert(ann.metadata.contains("summaryTokensEst"), "summaryTokensEst missing")
  }

  protected def summarizer(method: String): Summarization =
    new Summarization().setInputCols("document").setOutputCol("summary").setMethod(method)
}
