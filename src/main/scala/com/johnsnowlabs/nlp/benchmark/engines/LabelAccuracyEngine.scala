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
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD

/** Backs the per-item label-accuracy tasks (POS, Classification, SpellCheck, LanguageDetection)
  * and top-k label accuracy (ImageClassification), via Spark's own distributed
  * `org.apache.spark.mllib.evaluation.MulticlassMetrics` rather than a driver-collected loop,
  * following the precedent in `Doc2VecTestSpec`.
  */
object LabelAccuracyEngine {

  /** @param pairs
    *   (predictedLabel, goldLabel) per scored item, distributed
    */
  def evaluate(task: BenchmarkTask, pairs: RDD[(String, String)]): AccuracyReport = {
    val indexed = indexPairs(pairs)
    if (indexed._2.isEmpty) AccuracyReport(task, Map.empty, Map.empty, 0L)
    else report(task, indexed)
  }

  private val NoPrediction = "<<no_prediction>>"

  /** Scores the pipeline's own top-1 label the same way [[evaluate]] does, and reports top-k as
    * an additional `top${k}Accuracy` metric alongside it.
    *
    * @param pairs
    *   (topKPredictedLabels, goldLabel) per scored item, best-first
    */
  def evaluateTopK(
      task: BenchmarkTask,
      pairs: RDD[(Seq[String], String)],
      k: Int): AccuracyReport = {
    val scored = pairs.map { case (topK, gold) =>
      (topK.headOption.getOrElse(NoPrediction), gold, if (topK.take(k).contains(gold)) 1L else 0L)
    }
    scored.persist()
    try {
      val (hits, n) = scored
        .map(r => (r._3, 1L))
        .fold((0L, 0L)) { case ((h1, n1), (h2, n2)) => (h1 + h2, n1 + n2) }
      val base = evaluate(task, scored.map(r => (r._1, r._2)))
      val topKAccuracy = if (n == 0) 0.0 else hits.toDouble / n
      base.copy(overall = base.overall + (s"top${k}Accuracy" -> topKAccuracy))
    } finally scored.unpersist()
  }

  private def indexPairs(
      pairs: RDD[(String, String)]): (RDD[(Double, Double)], IndexedSeq[String]) = {
    val labels = pairs
      .flatMap { case (p, g) => Seq(p, g) }
      .distinct()
      .collect()
      .sorted
      .toIndexedSeq
    val labelIndex = labels.zipWithIndex.toMap
    val indexed = pairs.map { case (p, g) => (labelIndex(p).toDouble, labelIndex(g).toDouble) }
    (indexed, labels)
  }

  private def report(
      task: BenchmarkTask,
      indexed: (RDD[(Double, Double)], IndexedSeq[String])): AccuracyReport = {
    val (predictionAndLabels, labels) = indexed
    predictionAndLabels.persist()
    val support = predictionAndLabels.count()
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // "weighted" here means support-weighted across labels (Spark MLlib's convention), NOT the
    // pooled/micro average `SpanF1Engine`'s `overall` reports for NER/WordSegmentation, and NOT
    // the macro average (`ClassifierMetrics`/`TensorflowNer`) a model already prints during its
    // own training-time evaluation. All three are standard, legitimate choices for their
    // respective domains -- do not assume this "overall" is comparable across tasks, or to a
    // number the same model reported during training.
    val overall = Map(
      "accuracy" -> metrics.accuracy,
      "weightedPrecision" -> metrics.weightedPrecision,
      "weightedRecall" -> metrics.weightedRecall,
      "weightedF1" -> metrics.weightedFMeasure)

    // MulticlassMetrics throws for a label with no predictions or no gold occurrences.
    def safeMetric(f: Double => Double, label: Double): Double =
      try f(label)
      catch { case _: NoSuchElementException => 0.0 }

    val perClass = labels.zipWithIndex.map { case (label, idx) =>
      val d = idx.toDouble
      label -> Map(
        "precision" -> safeMetric(metrics.precision, d),
        "recall" -> safeMetric(metrics.recall, d),
        "f1" -> safeMetric(metrics.fMeasure, d))
    }.toMap

    predictionAndLabels.unpersist()
    AccuracyReport(task, overall, perClass, support)
  }
}
