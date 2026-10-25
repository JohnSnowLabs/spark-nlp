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

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.DefaultPragmaticMethod

/** Pure (Spark-free) building blocks for the [[Summarization]] annotator.
  *
  * Contains:
  *   - a rule-based sentence splitter (reusing the pragmatic sentence detector engine),
  *   - a sentence packer that groups sentences into character-budgeted chunks with overlap (used
  *     for hierarchical long-document summarization),
  *   - a PacSum-style position-augmented centrality ranker with MMR selection (used by the
  *     `extractive` method).
  *
  * All functions are deterministic and side-effect free so they can be used inside Spark UDFs and
  * unit-tested without a SparkSession.
  */
private[seq2seq] object ExtractiveSummarizationRanker {

  /** Splits text into sentences using the rule-based pragmatic method (no model download). */
  def splitSentences(text: String): Array[String] = {
    if (text == null || text.trim.isEmpty) return Array.empty[String]
    val method = new DefaultPragmaticMethod(useAbbreviations = true, detectLists = false)
    val bounds = method.extractBounds(text)
    val sentences = bounds.map(_.content.trim).filter(_.nonEmpty)
    if (sentences.isEmpty) Array(text.trim) else sentences
  }

  /** Packs sentences into chunks whose length stays under `budgetChars`, with `overlapSentences`
    * sentences repeated between consecutive chunks for context continuity.
    *
    * A single sentence longer than the budget is hard-truncated to the budget.
    */
  def packChunks(
      sentences: Array[String],
      budgetChars: Int,
      overlapSentences: Int): Array[String] = {
    require(budgetChars > 0, "budgetChars must be > 0")
    if (sentences.isEmpty) return Array.empty[String]

    val chunks = scala.collection.mutable.ArrayBuffer.empty[String]
    val current = scala.collection.mutable.ArrayBuffer.empty[String]
    var currentLen = 0

    def flush(): Unit = {
      if (current.nonEmpty) {
        chunks += current.mkString(" ")
        val overlap = current.takeRight(math.max(0, overlapSentences)).toArray
        current.clear()
        currentLen = 0
        // seed next chunk with the overlap sentences (never let overlap alone exceed budget)
        overlap.foreach { s =>
          if (currentLen + s.length + 1 <= budgetChars / 2) {
            current += s
            currentLen += s.length + 1
          }
        }
      }
    }

    sentences.foreach { raw =>
      val s = if (raw.length > budgetChars) raw.substring(0, budgetChars) else raw
      if (currentLen + s.length + 1 > budgetChars && current.nonEmpty) flush()
      current += s
      currentLen += s.length + 1
    }
    if (current.nonEmpty) chunks += current.mkString(" ")

    chunks.toArray
  }

  private def dot(a: Array[Float], b: Array[Float]): Double = {
    var s = 0.0
    var i = 0
    val n = math.min(a.length, b.length)
    while (i < n) { s += a(i).toDouble * b(i).toDouble; i += 1 }
    s
  }

  private def norm(a: Array[Float]): Double = math.sqrt(dot(a, a))

  /** Cosine similarity; 0.0 when either vector is degenerate. */
  def cosine(a: Array[Float], b: Array[Float]): Double = {
    val na = norm(a)
    val nb = norm(b)
    if (na == 0.0 || nb == 0.0) 0.0 else dot(a, b) / (na * nb)
  }

  /** PacSum-style position-augmented asymmetric degree centrality.
    *
    * `score(i) = sum_{j>i} sim(i,j) * wForward + sum_{j<i} sim(i,j) * wBackward + positionBias /
    * (1 + i)`
    *
    * Forward edges (similarity to *later* sentences) dominate, giving earlier sentences that
    * introduce recurring content higher centrality; the explicit position prior encodes the lead
    * bias that strong extractive baselines rely on.
    */
  def centralityScores(
      embeddings: Array[Array[Float]],
      wForward: Double = 1.0,
      wBackward: Double = 0.3,
      positionBias: Double = 0.3): Array[Double] = {
    val n = embeddings.length
    val scores = new Array[Double](n)
    if (n == 0) return scores
    if (n == 1) return Array(1.0 + positionBias)

    // precompute pairwise cosine (upper triangle)
    var i = 0
    while (i < n) {
      var j = i + 1
      while (j < n) {
        val sim = math.max(0.0, cosine(embeddings(i), embeddings(j)))
        scores(i) += sim * wForward // i sees a later sentence j
        scores(j) += sim * wBackward // j sees an earlier sentence i
        j += 1
      }
      i += 1
    }
    // normalize centrality to [0,1] before adding the position prior so the prior's scale
    // is comparable across documents of different lengths
    val max = scores.max
    i = 0
    while (i < n) {
      val centrality = if (max > 0.0) scores(i) / max else 0.0
      scores(i) = centrality + positionBias / (1.0 + i)
      i += 1
    }
    scores
  }

  /** Greedy MMR selection under a word budget.
    *
    * Picks `argmax_i lambda * score(i) - (1 - lambda) * maxSim(i, selected)` until adding the
    * next sentence would exceed `budgetWords` (at least one sentence is always selected).
    *
    * @return
    *   selected sentence indices in original document order
    */
  def mmrSelect(
      embeddings: Array[Array[Float]],
      scores: Array[Double],
      wordCounts: Array[Int],
      budgetWords: Int,
      lambda: Double): Array[Int] = {
    val n = embeddings.length
    if (n == 0) return Array.empty[Int]

    val selected = scala.collection.mutable.ArrayBuffer.empty[Int]
    val remaining = scala.collection.mutable.LinkedHashSet(0 until n: _*)
    var usedWords = 0

    while (remaining.nonEmpty) {
      var bestIdx = -1
      var bestVal = Double.MinValue
      remaining.foreach { i =>
        val redundancy =
          if (selected.isEmpty) 0.0
          else selected.map(j => math.max(0.0, cosine(embeddings(i), embeddings(j)))).max
        val value = lambda * scores(i) - (1.0 - lambda) * redundancy
        if (value > bestVal) {
          bestVal = value
          bestIdx = i
        }
      }
      if (bestIdx < 0) return selected.sorted.toArray

      val cost = math.max(1, wordCounts(bestIdx))
      if (selected.nonEmpty && usedWords + cost > budgetWords) {
        // budget exhausted
        return selected.sorted.toArray
      }
      selected += bestIdx
      usedWords += cost
      remaining -= bestIdx
      if (usedWords >= budgetWords) return selected.sorted.toArray
    }
    selected.sorted.toArray
  }

  def countWords(s: String): Int =
    if (s == null || s.trim.isEmpty) 0 else s.trim.split("\\s+").length
}
