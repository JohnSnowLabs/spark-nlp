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

package com.johnsnowlabs.nlp.benchmark.engines

import com.johnsnowlabs.nlp.benchmark.{AccuracyReport, BenchmarkTask}
import org.apache.spark.rdd.RDD

/** Backs dependency parsing: UAS (unlabeled attachment score) and LAS (labeled attachment score),
  * computed as a distributed map-then-reduce over per-token (head, label) pairs.
  */
object DependencyAccuracyEngine {

  /** @param rows
    *   (predicted, gold) per sentence, each a same-length sequence of (headIndex,
    *   dependencyLabel) per token
    */
  def evaluate(
      task: BenchmarkTask,
      rows: RDD[(Seq[(Int, String)], Seq[(Int, String)])]): AccuracyReport = {
    val taskName = task.name
    val (correctHead, correctHeadAndLabel, total) = rows
      .map { case (predicted, gold) =>
        require(
          predicted.length == gold.length,
          s"Benchmark.evaluate(task = $taskName): predicted and gold sequences have " +
            s"different lengths (${predicted.length} vs ${gold.length}) for one row. labelCol " +
            "must align one-to-one with the pipeline's own tokenization for this task.")
        predicted.zip(gold).foldLeft((0L, 0L, 0L)) {
          case ((ch, chl, n), ((pHead, pLabel), (gHead, gLabel))) =>
            val headMatch = if (pHead == gHead) 1L else 0L
            val bothMatch = if (pHead == gHead && pLabel == gLabel) 1L else 0L
            (ch + headMatch, chl + bothMatch, n + 1L)
        }
      }
      .fold((0L, 0L, 0L)) { case ((ch1, chl1, n1), (ch2, chl2, n2)) =>
        (ch1 + ch2, chl1 + chl2, n1 + n2)
      }

    val uas = if (total == 0) 0.0 else correctHead.toDouble / total
    val las = if (total == 0) 0.0 else correctHeadAndLabel.toDouble / total

    AccuracyReport(task, Map("uas" -> uas, "las" -> las), support = total)
  }
}
