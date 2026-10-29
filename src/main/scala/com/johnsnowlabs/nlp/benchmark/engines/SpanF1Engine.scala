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

/** Backs NER and word segmentation: standard `conlleval`-style entity/segment-level
  * precision/recall/F1, computed as a distributed map (each row to local TP/FP/FN counts) then
  * reduce, rather than collecting tag sequences to the driver.
  */
object SpanF1Engine {

  private[engines] case class Span(start: Int, end: Int, entityType: String)

  // "E"/"L" (IOBES "end"/BILOU "last") and "S"/"U" (IOBES "single"/BILOU "unit") are aliased onto
  // a shared "E"/"S" bucket below, so extractSpans handles all three tagging schemes (BIO/IOB2,
  // IOBES, BILOU) uniformly.
  private[engines] def splitTag(tag: String): (String, String) = {
    if (tag == null || tag == "O") ("O", null)
    else {
      val dash = tag.indexOf('-')
      if (dash < 0) ("B", tag)
      else {
        val prefix = tag.substring(0, dash) match {
          case "L" => "E"
          case "U" => "S"
          case p => p
        }
        (prefix, tag.substring(dash + 1))
      }
    }
  }

  private[engines] def extractSpans(tags: Seq[String]): Set[Span] = {
    val spans = scala.collection.mutable.ArrayBuffer[Span]()
    var start = -1
    var currentType: String = null

    def closeSpan(endExclusive: Int): Unit = {
      if (start >= 0) spans += Span(start, endExclusive, currentType)
      start = -1
      currentType = null
    }

    tags.zipWithIndex.foreach { case (tag, i) =>
      splitTag(tag) match {
        case ("O", _) => closeSpan(i)
        case ("B", t) =>
          closeSpan(i)
          start = i
          currentType = t
        case ("I", t) =>
          if (start < 0 || currentType != t) {
            closeSpan(i)
            start = i
            currentType = t
          }
        // "E-X"/"L-X": closes a chunk inclusive of this token -- extends an already-open same-type
        // span if there is one, otherwise (malformed input, a type change mid-span, or a genuine
        // IOBES/BILOU-style single-token close with no preceding B-/I-) closes whatever was open
        // first (so it isn't silently dropped) before opening a one-token span right here.
        case ("E", t) =>
          if (start < 0 || currentType != t) {
            closeSpan(i)
            start = i
            currentType = t
          }
          closeSpan(i + 1)
        // "S-X"/"U-X": always a standalone single-token entity, regardless of any span already
        // open (which gets closed first, unaffected by this one).
        case ("S", t) =>
          closeSpan(i)
          spans += Span(i, i + 1, t)
        case _ => closeSpan(i)
      }
    }
    closeSpan(tags.length)
    spans.toSet
  }

  private type Counts = (Long, Long, Long) // (tp, fp, fn)

  private def mergeCounts(a: Map[String, Counts], b: Map[String, Counts]): Map[String, Counts] = {
    (a.keySet ++ b.keySet).map { k =>
      val (tp1, fp1, fn1) = a.getOrElse(k, (0L, 0L, 0L))
      val (tp2, fp2, fn2) = b.getOrElse(k, (0L, 0L, 0L))
      k -> (tp1 + tp2, fp1 + fp2, fn1 + fn2)
    }.toMap
  }

  private def prf(counts: Counts): Map[String, Double] = {
    val (tp, fp, fn) = counts
    val precision = if (tp + fp == 0) 0.0 else tp.toDouble / (tp + fp)
    val recall = if (tp + fn == 0) 0.0 else tp.toDouble / (tp + fn)
    val f1 = if (precision + recall == 0) 0.0 else 2 * precision * recall / (precision + recall)
    Map("precision" -> precision, "recall" -> recall, "f1" -> f1)
  }

  /** @param rows
    *   (predictedTags, goldTags) per document/sentence, BIO/IOB2-tagged, same length within a row
    */
  def evaluate(task: BenchmarkTask, rows: RDD[(Seq[String], Seq[String])]): AccuracyReport =
    scoreSpans(
      task,
      rows.map { case (predTags, goldTags) =>
        (extractSpans(predTags), extractSpans(goldTags))
      })

  /** For tasks where spans are already known boundaries rather than something to derive from BIO
    * tags (e.g. word segmentation, where each output token's (begin, end) offset is already a
    * segment). All spans are scored as a single implicit type.
    *
    * @param rows
    *   (predictedBoundaries, goldBoundaries) per document/sentence, each a set of (begin, end)
    *   offsets
    */
  def evaluateBoundaries(
      task: BenchmarkTask,
      rows: RDD[(Set[(Int, Int)], Set[(Int, Int)])]): AccuracyReport =
    scoreSpans(
      task,
      rows.map { case (pred, gold) =>
        (
          pred.map { case (s, e) => Span(s, e, "segment") },
          gold.map { case (s, e) => Span(s, e, "segment") })
      })

  private def scoreSpans(
      task: BenchmarkTask,
      rows: RDD[(Set[Span], Set[Span])]): AccuracyReport = {
    val perRowCounts: RDD[Map[String, Counts]] = rows.map { case (predSpans, goldSpans) =>
      val types = predSpans.map(_.entityType) ++ goldSpans.map(_.entityType)
      types.map { t =>
        val predT = predSpans.filter(_.entityType == t)
        val goldT = goldSpans.filter(_.entityType == t)
        val tp = predT.intersect(goldT).size.toLong
        t -> (tp, predT.size - tp, goldT.size - tp)
      }.toMap
    }

    val totals = perRowCounts.fold(Map.empty[String, Counts])(mergeCounts)

    // Micro-averaged: true/false positives and false negatives are pooled across every entity
    // type before computing one precision/recall/F1, matching conlleval/seqeval's standard
    // convention for span-level NER scoring. This is NOT the same averaging convention as
    // `LabelAccuracyEngine`'s support-weighted "overall" -- don't assume the two are comparable --
    // and exact span-boundary matching means this will typically differ from the per-token tag
    // accuracy a model already reports during its own training-time evaluation
    // (`TensorflowNer.measure`).
    val overallCounts = totals.values.foldLeft((0L, 0L, 0L)) { case ((tp, fp, fn), (t, f, n)) =>
      (tp + t, fp + f, fn + n)
    }
    val support = totals.values.map { case (tp, _, fn) => tp + fn }.sum

    AccuracyReport(task, prf(overallCounts), totals.map { case (t, c) => t -> prf(c) }, support)
  }
}
