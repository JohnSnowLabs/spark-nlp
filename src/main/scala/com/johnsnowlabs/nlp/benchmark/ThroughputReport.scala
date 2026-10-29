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

/** Throughput of a single output annotator type, averaged over the timed trials.
  *
  * @param annotatorType
  *   the `annotatorType` of the output column this rate was computed from (e.g. `token`,
  *   `named_entity`)
  * @param outputColumn
  *   the name of the output column
  * @param totalItems
  *   total annotations produced for this column across all timed trials
  * @param meanItemsPerSecond
  *   mean rate across trials
  * @param confidenceInterval95
  *   half-width of the 95% confidence interval around [[meanItemsPerSecond]] (normal
  *   approximation); `0.0` when there is only one trial
  */
case class MetricRate(
    annotatorType: String,
    outputColumn: String,
    totalItems: Long,
    meanItemsPerSecond: Double,
    confidenceInterval95: Double) {

  override def toString: String =
    // Leads with outputColumn, not annotatorType: SentenceDetector's output and
    // DocumentAssembler's are both annotatorType "document", so annotatorType alone can't
    // tell two rows in the same report apart (e.g. a "document" column and a "sentence"
    // column both showing as "document").
    f"$outputColumn%-20s ${meanItemsPerSecond}%,.1f ± ${confidenceInterval95}%,.1f items/sec (type: $annotatorType, n=$totalItems)"
}

/** Result of [[Benchmark.throughput]]: one [[MetricRate]] per annotation type the pipeline
  * produced, plus the raw per-trial elapsed times behind them.
  */
case class ThroughputReport(rates: Seq[MetricRate], trialSeconds: Seq[Double]) {

  override def toString: String = {
    val header = f"Throughput over ${trialSeconds.length} trial(s), " +
      f"mean elapsed ${trialSeconds.sum / trialSeconds.length}%,.3f sec/trial"
    (header +: rates.map(r => "  " + r.toString)).mkString("\n")
  }
}
