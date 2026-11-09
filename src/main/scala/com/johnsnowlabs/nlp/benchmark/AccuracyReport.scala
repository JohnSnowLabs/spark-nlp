/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.benchmark

/** Result of [[Benchmark.evaluate]]. The metric names in [[overall]] and [[perClass]] depend on
  * the [[BenchmarkTask]]: label-accuracy tasks report `accuracy`/`weightedPrecision`/
  * `weightedRecall`/`weightedF1` (support-weighted across labels); NER and word segmentation
  * report entity/segment-level `precision`/`recall`/`f1` (micro-averaged: pooled counts across
  * every type, matching conlleval/seqeval, via exact span-boundary matching); dependency parsing
  * reports `uas`/`las`; the text-similarity tasks report their own metric (`bleu`,
  * `rouge1`/`rouge2`/`rougeL`, `wer`, or `exactMatch`/`f1`).
  *
  * These averaging conventions differ by task and are NOT interchangeable: `overall("f1")` from a
  * `Classification` report and from a `NER` report are computed differently and are not
  * comparable to each other. They can also differ from a metric the same model already reported
  * during its own training-time evaluation (which typically uses per-token tag accuracy for NER,
  * or macro-averaged F1 for classification) -- that is expected, not a discrepancy to reconcile.
  *
  * @param task
  *   the task this report was scored against
  * @param overall
  *   corpus-level metric name -> value
  * @param perClass
  *   per-label/per-entity-type breakdown, metric name -> value; empty for tasks where a per-class
  *   breakdown doesn't apply (e.g. the text-similarity tasks)
  * @param support
  *   number of scored items (tokens, spans, or documents, depending on the task)
  * @param scoredColumns
  *   which of the pipeline's output columns this score was actually computed from
  */
case class AccuracyReport(
    task: BenchmarkTask,
    overall: Map[String, Double],
    perClass: Map[String, Map[String, Double]] = Map.empty,
    support: Long = 0L,
    scoredColumns: Seq[String] = Seq.empty) {

  override def toString: String = {
    val overallLine = overall.toSeq
      .sortBy(_._1)
      .map { case (k, v) => f"$k=$v%.4f" }
      .mkString(", ")
    val scored =
      if (scoredColumns.isEmpty) "" else s", scored: ${scoredColumns.mkString(", ")}"
    val header = s"${task.name} accuracy (n=$support$scored): $overallLine"
    if (perClass.isEmpty) header
    else {
      val sorted = perClass.toSeq.sortBy(_._1)
      val classLines = sorted
        .take(AccuracyReport.MaxPrintedClasses)
        .map { case (label, metrics) =>
          val m = metrics.toSeq.sortBy(_._1).map { case (k, v) => f"$k=$v%.4f" }.mkString(", ")
          s"  $label: $m"
        }
      val elided =
        if (sorted.length <= AccuracyReport.MaxPrintedClasses) Seq.empty
        else
          Seq(
            s"  ... and ${sorted.length - AccuracyReport.MaxPrintedClasses} more labels " +
              "(see the perClass field for the full breakdown)")
      (header +: (classLines ++ elided)).mkString("\n")
    }
  }
}

object AccuracyReport {

  /** Cap on how many per-class lines [[AccuracyReport.toString]] prints before eliding. */
  val MaxPrintedClasses: Int = 50
}
