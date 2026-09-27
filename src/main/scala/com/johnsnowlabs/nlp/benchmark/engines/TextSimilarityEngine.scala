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

sealed trait TextMetric
object TextMetric {
  case object Bleu extends TextMetric
  case object Rouge extends TextMetric
  case object Wer extends TextMetric
  case object SquadEmF1 extends TextMetric
}

/** Backs the generated-text tasks (Translation/BLEU, Summarization/ROUGE, SpeechRecognition/WER,
  * QuestionAnswering/SQuAD EM+F1). No JVM equivalents of `sacrebleu`/`rouge_score`/`jiwer` exist,
  * so these are original implementations matching those packages' documented default behavior.
  */
object TextSimilarityEngine {

  // sacrebleu's `TokenizerRegexp` character class.
  private val Punct13a = "([\\x7B-\\x7E\\x5B-\\x60\\x20-\\x26\\x28-\\x2B\\x3A-\\x40\\x2F])"

  // Port of sacrebleu's default `13a` tokenizer. Keeps apostrophes attached to their word
  // (`l'homme` stays one token) and a period/comma between two digits (`3.14`) intact.
  private def bleuTokenize(text: String): Array[String] = {
    var line = text
      .replace("<skipped>", "")
      .replace("-\n", "")
      .replace("\n", " ")
    if (line.contains("&")) {
      line = line
        .replace("&quot;", "\"")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    }
    line = " " + line + " "
    line = line.replaceAll(Punct13a, " $1 ")
    // period/comma split only when not flanked by a digit on the relevant side
    line = line.replaceAll("([^0-9])([\\.,])", "$1 $2 ")
    line = line.replaceAll("([\\.,])([^0-9])", " $1 $2")
    // hyphen split only when preceded by a digit
    line = line.replaceAll("([0-9])(-)", "$1 $2 ")
    line.split("\\s+").filter(_.nonEmpty)
  }

  // jiwer's default WER tokenization is plain whitespace splitting.
  private def werTokenize(text: String): Array[String] =
    text.trim.split("\\s+").filter(_.nonEmpty)

  // Approximates rouge-score's default tokenizer (lowercase, strip non-alphanumeric).
  private def rougeTokenize(text: String): Array[String] =
    text.toLowerCase.replaceAll("[^a-z0-9\\s]", " ").split("\\s+").filter(_.nonEmpty)

  // Exact port of the official SQuAD eval script's `normalize_answer`.
  private def squadNormalize(text: String): String = {
    val lower = text.toLowerCase
    val noPunct = lower.filterNot(c => "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".contains(c))
    val noArticles = noPunct.replaceAll("\\b(a|an|the)\\b", " ")
    noArticles.trim.split("\\s+").filter(_.nonEmpty).mkString(" ")
  }

  private def ngrams(tokens: Seq[String], n: Int): Map[String, Int] =
    tokens
      .sliding(n)
      .filter(_.length == n)
      .map(_.mkString(" "))
      .toSeq
      .groupBy(identity)
      .mapValues(_.size)
      .toMap

  private def levenshtein(a: Array[String], b: Array[String]): Int = {
    val dp = Array.tabulate(a.length + 1, b.length + 1) { (i, j) =>
      if (i == 0) j else if (j == 0) i else 0
    }
    for (i <- 1 to a.length; j <- 1 to b.length) {
      dp(i)(j) =
        if (a(i - 1) == b(j - 1)) dp(i - 1)(j - 1)
        else 1 + math.min(dp(i - 1)(j - 1), math.min(dp(i - 1)(j), dp(i)(j - 1)))
    }
    dp(a.length)(b.length)
  }

  private def lcsLength(a: Array[String], b: Array[String]): Int = {
    val dp = Array.ofDim[Int](a.length + 1, b.length + 1)
    for (i <- 1 to a.length; j <- 1 to b.length) {
      dp(i)(j) =
        if (a(i - 1) == b(j - 1)) dp(i - 1)(j - 1) + 1 else math.max(dp(i - 1)(j), dp(i)(j - 1))
    }
    dp(a.length)(b.length)
  }

  private def prf(overlap: Double, predLen: Double, goldLen: Double): (Double, Double, Double) = {
    val precision = if (predLen == 0) 0.0 else overlap / predLen
    val recall = if (goldLen == 0) 0.0 else overlap / goldLen
    val f1 = if (precision + recall == 0) 0.0 else 2 * precision * recall / (precision + recall)
    (precision, recall, f1)
  }

  private case class BleuAcc(
      matched: Array[Long],
      total: Array[Long],
      hypLen: Long,
      refLen: Long,
      rows: Long)

  private def bleu(pairs: RDD[(String, String)]): (Map[String, Double], Long) = {
    val perRow = pairs.map { case (pred, gold) =>
      val hyp = bleuTokenize(pred)
      val ref = bleuTokenize(gold)
      val matched = Array.fill(4)(0L)
      val total = Array.fill(4)(0L)
      for (n <- 1 to 4) {
        val hypNgrams = ngrams(hyp, n)
        val refNgrams = ngrams(ref, n)
        matched(n - 1) =
          hypNgrams.map { case (g, c) => math.min(c, refNgrams.getOrElse(g, 0)) }.sum.toLong
        total(n - 1) = hypNgrams.values.sum.toLong
      }
      BleuAcc(matched, total, hyp.length.toLong, ref.length.toLong, 1L)
    }
    val zero = BleuAcc(Array.fill(4)(0L), Array.fill(4)(0L), 0L, 0L, 0L)
    val totals = perRow.fold(zero) { (a, b) =>
      BleuAcc(
        Array.tabulate(4)(i => a.matched(i) + b.matched(i)),
        Array.tabulate(4)(i => a.total(i) + b.total(i)),
        a.hypLen + b.hypLen,
        a.refLen + b.refLen,
        a.rows + b.rows)
    }

    // Ports sacrebleu's default `smooth_method='exp'` (NIST) precisely. Verified directly against
    // real sacrebleu: when the corpus (pooled across every row, not per-sentence) has zero total
    // n-grams of some order -- every hypothesis/reference combined is shorter than that order --
    // corpus-level BLEU is 0 by design, even for an otherwise-perfect match. This is standard BLEU
    // behavior, not a bug: `sacrebleu.corpus_bleu(["3.14 is pi"], [["3.14 is pi"]]).score == 0.0`,
    // while mixing in even one longer sentence changes the corpus-level 4-gram total to nonzero
    // and the score reflects the match normally. Do not "fix" this into an effective-order
    // reduction (sacrebleu only does that for per-sentence scoring with `effective_order=True`,
    // not for corpus_bleu's default).
    val precisions = Array.fill(4)(0.0)
    // sacrebleu has an explicit early-return (https://github.com/mjpost/sacrebleu/issues/141):
    // when the hypothesis shares zero n-grams with the reference at *every* order, every
    // precision is forced to exactly 0 before smoothing ever runs -- smoothing individual orders
    // in that case (as the loop below does when at least one order did match) produces nonzero
    // "precisionN" diagnostic values sacrebleu itself would never report. The headline `bleuScore`
    // below already special-cases this via its own `matched.forall(_ == 0L)` guard; this mirrors
    // that guard onto the per-order fields it's built from.
    if (!totals.matched.forall(_ == 0L)) {
      var smoothMteval = 1.0
      var brokeEarly = false
      for (i <- 0 until 4) {
        if (!brokeEarly) {
          if (totals.total(i) == 0) brokeEarly = true
          else if (totals.matched(i) == 0) {
            smoothMteval *= 2
            precisions(i) = 1.0 / (smoothMteval * totals.total(i))
          } else precisions(i) = totals.matched(i).toDouble / totals.total(i)
        }
      }
    }
    val brevityPenalty =
      if (totals.hypLen >= totals.refLen) 1.0
      else if (totals.hypLen == 0) 0.0
      else math.exp(1.0 - totals.refLen.toDouble / totals.hypLen)
    val bleuScore =
      if (totals.matched.forall(_ == 0L)) 0.0
      else brevityPenalty * math.exp(precisions.map(math.log).sum / 4.0)

    (
      Map(
        "bleu" -> bleuScore,
        "precision1" -> precisions(0),
        "precision2" -> precisions(1),
        "precision3" -> precisions(2),
        "precision4" -> precisions(3),
        "brevityPenalty" -> brevityPenalty),
      totals.rows)
  }

  private case class RougeRowScores(
      r1p: Double,
      r1r: Double,
      r1f: Double,
      r2p: Double,
      r2r: Double,
      r2f: Double,
      rLp: Double,
      rLr: Double,
      rLf: Double,
      rows: Long)

  private def rouge(pairs: RDD[(String, String)]): (Map[String, Double], Long) = {
    val perRow = pairs.map { case (pred, gold) =>
      val hyp = rougeTokenize(pred)
      val ref = rougeTokenize(gold)

      def ngramOverlap(n: Int): (Double, Double, Double) = {
        val hypNgrams = ngrams(hyp, n)
        val refNgrams = ngrams(ref, n)
        val overlap =
          hypNgrams.map { case (g, c) => math.min(c, refNgrams.getOrElse(g, 0)) }.sum.toDouble
        prf(overlap, hypNgrams.values.sum.toDouble, refNgrams.values.sum.toDouble)
      }

      val (r1p, r1r, r1f) = ngramOverlap(1)
      val (r2p, r2r, r2f) = ngramOverlap(2)
      val lcs = lcsLength(hyp, ref).toDouble
      val (rLp, rLr, rLf) = prf(lcs, hyp.length.toDouble, ref.length.toDouble)
      RougeRowScores(r1p, r1r, r1f, r2p, r2r, r2f, rLp, rLr, rLf, 1L)
    }

    val sum = perRow.fold(RougeRowScores(0, 0, 0, 0, 0, 0, 0, 0, 0, 0L)) { (a, b) =>
      RougeRowScores(
        a.r1p + b.r1p,
        a.r1r + b.r1r,
        a.r1f + b.r1f,
        a.r2p + b.r2p,
        a.r2r + b.r2r,
        a.r2f + b.r2f,
        a.rLp + b.rLp,
        a.rLr + b.rLr,
        a.rLf + b.rLf,
        a.rows + b.rows)
    }
    val n = sum.rows
    if (n == 0) (Map.empty, 0L)
    else {
      (
        Map(
          "rouge1_precision" -> sum.r1p / n,
          "rouge1_recall" -> sum.r1r / n,
          "rouge1_f1" -> sum.r1f / n,
          "rouge2_precision" -> sum.r2p / n,
          "rouge2_recall" -> sum.r2r / n,
          "rouge2_f1" -> sum.r2f / n,
          "rougeL_precision" -> sum.rLp / n,
          "rougeL_recall" -> sum.rLr / n,
          "rougeL_f1" -> sum.rLf / n),
        n)
    }
  }

  private def wer(pairs: RDD[(String, String)]): (Map[String, Double], Long) = {
    val perRow = pairs.map { case (pred, gold) =>
      val hyp = werTokenize(pred)
      val ref = werTokenize(gold)
      (levenshtein(hyp, ref).toLong, ref.length.toLong, 1L)
    }
    val (edits, refWords, rows) = perRow.fold((0L, 0L, 0L)) { case ((e1, r1, n1), (e2, r2, n2)) =>
      (e1 + e2, r1 + r2, n1 + n2)
    }
    val werScore = if (refWords == 0) 0.0 else edits.toDouble / refWords
    (Map("wer" -> werScore), rows)
  }

  private def squadEmF1(pairs: RDD[(String, Seq[String])]): (Map[String, Double], Long) = {
    val perRow = pairs.map { case (pred, golds) =>
      val hyp = squadNormalize(pred)
      val hypTokens = hyp.split("\\s+").filter(_.nonEmpty)
      val hypCounts = hypTokens.groupBy(identity).mapValues(_.length).toMap

      // Scores against every reference answer and keeps the best, per the official SQuAD eval.
      val scored = golds.map { gold =>
        val ref = squadNormalize(gold)
        val em = if (hyp == ref) 1.0 else 0.0
        val refTokens = ref.split("\\s+").filter(_.nonEmpty)
        val refCounts = refTokens.groupBy(identity).mapValues(_.length).toMap
        val overlap =
          hypCounts.map { case (t, c) => math.min(c, refCounts.getOrElse(t, 0)) }.sum.toDouble
        val (_, _, f1) = prf(overlap, hypTokens.length.toDouble, refTokens.length.toDouble)
        (em, f1)
      }

      val em = if (scored.isEmpty) 0.0 else scored.map(_._1).max
      val f1 = if (scored.isEmpty) 0.0 else scored.map(_._2).max
      (em, f1, 1L)
    }
    val (emSum, f1Sum, n) = perRow.fold((0.0, 0.0, 0L)) { case ((e1, f1a, n1), (e2, f1b, n2)) =>
      (e1 + e2, f1a + f1b, n1 + n2)
    }
    if (n == 0) (Map.empty, 0L)
    else (Map("exactMatch" -> emSum / n, "f1" -> f1Sum / n), n)
  }

  /** @param pairs
    *   (generatedText, referenceText) per example
    */
  def evaluate(
      task: BenchmarkTask,
      pairs: RDD[(String, String)],
      metric: TextMetric): AccuracyReport = {
    val (overall, support) = metric match {
      case TextMetric.Bleu => bleu(pairs)
      case TextMetric.Rouge => rouge(pairs)
      case TextMetric.Wer => wer(pairs)
      case TextMetric.SquadEmF1 => squadEmF1(pairs.map { case (p, g) => (p, Seq(g)) })
    }
    AccuracyReport(task, overall, support = support)
  }

  /** SQuAD EM/F1 against several reference answers per question, scoring each prediction against
    * its best-matching reference the way the official SQuAD eval script does.
    *
    * @param pairs
    *   (predictedAnswer, allGoldAnswersForThatQuestion) per example
    */
  def evaluateSquad(task: BenchmarkTask, pairs: RDD[(String, Seq[String])]): AccuracyReport = {
    val (overall, support) = squadEmF1(pairs)
    AccuracyReport(task, overall, support = support)
  }
}
