/*
 * Copyright 2017-2024 John Snow Labs
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

package com.johnsnowlabs.ml.ai.util.Diarization

import java.nio.{ByteBuffer, ByteOrder}
import java.util.Base64
import scala.collection.mutable.ArrayBuffer

/** One detected, embedded speech turn awaiting a speaker label. */
case class SpeakerTurn(id: String, beginMs: Int, endMs: Int, embedding: Array[Float])

/** A turn after clustering: which speaker it was assigned to, and how confidently. */
case class ClusteredTurn(turn: SpeakerTurn, speakerLabel: String, confidence: Double)

/** One cluster's running centroid, carried across chunks/streaming calls so cross-chunk turns
  * merge into the same speaker instead of restarting the label space every chunk.
  *
  * @param label
  *   the speaker label this centroid currently represents
  * @param centroid
  *   running mean embedding of every turn merged into this cluster so far
  * @param memberCount
  *   number of turns merged into this cluster so far (used to weight the running mean)
  */
case class SpeakerCentroid(label: String, centroid: Array[Float], memberCount: Int)

/** Cluster state that can be threaded across `cluster` calls for the same session — either the
  * internal cross-chunk merge within one long file, or true multi-call streaming via the
  * serialized blob a caller threads back in (see `SpeakerDiarizer`'s streaming design).
  */
case class ClusterState(centroids: Seq[SpeakerCentroid], nextLabelIndex: Int)

object ClusterState {
  val empty: ClusterState = ClusterState(Seq.empty, nextLabelIndex = 0)
}

/** Deterministic, dependency-free speaker clustering over turn embeddings.
  *
  * Uses average-linkage agglomerative clustering over cosine distance. Deliberately avoids any
  * algorithm with random initialization (e.g. k-means): the same embeddings, in the same order,
  * always produce the same cluster assignments and the same label-to-cluster mapping — a
  * correctness requirement for speaker IDs that need to stay stable across repeated runs on the
  * same audio.
  */
object SpeakerClustering {

  /** 1 - cosine similarity. Both vectors are assumed non-zero (an all-zero embedding is a model
    * failure upstream, not something this function should silently tolerate).
    */
  def cosineDistance(a: Array[Float], b: Array[Float]): Double = {
    require(a.length == b.length, "Embeddings must have the same dimensionality")
    var dot = 0.0
    var normA = 0.0
    var normB = 0.0
    var i = 0
    while (i < a.length) {
      dot += a(i).toDouble * b(i).toDouble
      normA += a(i).toDouble * a(i).toDouble
      normB += b(i).toDouble * b(i).toDouble
      i += 1
    }
    val denom = math.sqrt(normA) * math.sqrt(normB)
    if (denom == 0.0) 1.0 else 1.0 - (dot / denom)
  }

  private def weightedAverage(
      a: Array[Float],
      weightA: Int,
      b: Array[Float],
      weightB: Int): Array[Float] = {
    val totalWeight = (weightA + weightB).toDouble
    Array.tabulate(a.length) { i =>
      ((a(i).toDouble * weightA + b(i).toDouble * weightB) / totalWeight).toFloat
    }
  }

  /** Clusters `turns` into speakers, optionally continuing from `priorState` (cross-chunk /
    * streaming continuation) and optionally renaming clusters to enrolled names via `gallery`.
    *
    * @param turns
    *   turns to cluster, each already embedded
    * @param numSpeakers
    *   if set, target exactly this many clusters. Whether that target is enforced by *splitting*
    *   apart natural clusters (not just merging them down) is controlled by `forceExactCount` —
    *   see that parameter, this is not always safe to do.
    * @param minSpeakers
    *   never merge below this many clusters (ignored if `numSpeakers` is set and enforced)
    * @param maxSpeakers
    *   never leave more than this many clusters unmerged, even if the closest pair exceeds
    *   `threshold` (ignored if `numSpeakers` is set and enforced)
    * @param threshold
    *   cosine-distance cutoff: merging stops once the closest remaining pair is farther than
    *   this, subject to `minSpeakers`/`maxSpeakers` (ignored if `numSpeakers` is set and
    *   enforced)
    * @param priorState
    *   cluster centroids carried in from a previous chunk/call, treated as additional points
    *   available to merge into — this is what makes speaker labels consistent across chunks
    * @param gallery
    *   enrolled name -> reference embedding; a final cluster centroid within
    *   `galleryAcceptanceDistance` of an entry is renamed to that entry's key
    * @param galleryAcceptanceDistance
    *   max cosine distance for a gallery match to be accepted
    * @param forceExactCount
    *   when `numSpeakers` is set and this is `true` (the default — correct for a single-shot call
    *   over a complete, fully-available recording), the exact count is enforced in both
    *   directions: merge down AND force-split apart via `minSpeakers`-style floor logic if fewer
    *   natural clusters exist than requested. When `false` (pass this for every chunk of a
    *   multi-chunk file except the last, and for every mid-session streaming call), `numSpeakers`
    *   only merges DOWN toward the target — it never fabricates an extra cluster out of noise to
    *   reach the count. Forcing the exact count before all speakers in a session have actually
    *   spoken creates a phantom cluster that then persists across chunks via `priorState` and
    *   competes with the real speaker who eventually appears — confirmed as a real failure mode
    *   during design review, not a hypothetical; this flag exists specifically to prevent it.
    * @return
    *   each input turn labeled with a speaker and a confidence, plus the resulting state to carry
    *   into the next chunk/call
    */
  def cluster(
      turns: Seq[SpeakerTurn],
      numSpeakers: Option[Int] = None,
      minSpeakers: Int = 1,
      maxSpeakers: Int = 20,
      threshold: Double = 0.7,
      priorState: ClusterState = ClusterState.empty,
      gallery: Map[String, Array[Float]] = Map.empty,
      galleryAcceptanceDistance: Double = 0.25,
      forceExactCount: Boolean = true): (Seq[ClusteredTurn], ClusterState) = {

    if (turns.isEmpty) return (Seq.empty, priorState)

    case class WorkingCluster(
        var label: String,
        var centroid: Array[Float],
        var memberCount: Int,
        var turnIndices: ArrayBuffer[Int])

    val clusters = new ArrayBuffer[WorkingCluster]()
    var nextLabelIndex = priorState.nextLabelIndex

    priorState.centroids.foreach { c =>
      clusters += WorkingCluster(c.label, c.centroid, c.memberCount, ArrayBuffer.empty)
    }
    turns.zipWithIndex.foreach { case (_, idx) =>
      clusters += WorkingCluster(
        label = null,
        centroid = turns(idx).embedding,
        memberCount = 1,
        turnIndices = ArrayBuffer(idx))
    }

    val distanceCache = Array.ofDim[Double](clusters.length, clusters.length)
    for (i <- clusters.indices; j <- (i + 1) until clusters.length) {
      val d = cosineDistance(clusters(i).centroid, clusters(j).centroid)
      distanceCache(i)(j) = d
      distanceCache(j)(i) = d
    }
    val alive = ArrayBuffer.fill(clusters.length)(true)

    def closestPair(): Option[(Int, Int, Double)] = {
      var best: Option[(Int, Int, Double)] = None
      var i = 0
      while (i < clusters.length) {
        if (alive(i)) {
          var j = i + 1
          while (j < clusters.length) {
            if (alive(j)) {
              val d = distanceCache(i)(j)
              val isBetter = best match {
                case None => true
                case Some((bi, bj, bd)) =>
                  d < bd || (d == bd && (i < bi || (i == bi && j < bj)))
              }
              if (isBetter) best = Some((i, j, d))
            }
            j += 1
          }
        }
        i += 1
      }
      best
    }

    def aliveCount: Int = alive.count(identity)

    def mergeStep(): Unit = {
      closestPair().foreach { case (i, j, _) =>
        val a = clusters(i)
        val b = clusters(j)
        a.centroid = weightedAverage(a.centroid, a.memberCount, b.centroid, b.memberCount)
        a.memberCount += b.memberCount
        a.turnIndices ++= b.turnIndices
        if (a.label == null) a.label = b.label
        alive(j) = false
        var k = 0
        while (k < clusters.length) {
          if (alive(k) && k != i) {
            val d = cosineDistance(a.centroid, clusters(k).centroid)
            distanceCache(i)(k) = d
            distanceCache(k)(i) = d
          }
          k += 1
        }
      }
    }

    def splitStep(): Boolean = {
      var bestClusterIdx = -1
      var bestTurnIdx = -1
      var bestDist = -1.0
      var i = 0
      while (i < clusters.length) {
        if (alive(i) && clusters(i).turnIndices.nonEmpty && clusters(i).memberCount > 1) {
          clusters(i).turnIndices.foreach { idx =>
            val d = cosineDistance(turns(idx).embedding, clusters(i).centroid)
            if (d > bestDist) {
              bestDist = d
              bestClusterIdx = i
              bestTurnIdx = idx
            }
          }
        }
        i += 1
      }
      if (bestClusterIdx == -1) false
      else {
        val c = clusters(bestClusterIdx)
        val removedEmbedding = turns(bestTurnIdx).embedding
        c.turnIndices -= bestTurnIdx
        val remainingWeight = c.memberCount - 1
        c.centroid = Array.tabulate(c.centroid.length) { d =>
          ((c.centroid(d).toDouble * c.memberCount - removedEmbedding(
            d).toDouble) / remainingWeight).toFloat
        }
        c.memberCount = remainingWeight
        clusters += WorkingCluster(
          label = null,
          centroid = removedEmbedding,
          memberCount = 1,
          turnIndices = ArrayBuffer(bestTurnIdx))
        alive += true
        true
      }
    }

    numSpeakers match {
      case Some(target) if forceExactCount =>
        val floor = math.max(1, target)
        while (aliveCount > floor && aliveCount > 1) mergeStep()
        while (aliveCount < floor && splitStep()) ()
      case Some(target) =>
        val cap = math.max(1, target)
        var continue = true
        while (continue && aliveCount > 1) {
          val shouldForceMerge = aliveCount > cap
          closestPair() match {
            case Some((_, _, d)) if shouldForceMerge || d <= threshold =>
              mergeStep()
            case _ => continue = false
          }
        }
      case None =>
        var continue = true
        while (continue && aliveCount > 1) {
          val shouldForceMerge = aliveCount > maxSpeakers
          closestPair() match {
            case Some((_, _, d))
                if shouldForceMerge || (d <= threshold && aliveCount > minSpeakers) =>
              mergeStep()
            case _ => continue = false
          }
        }
    }

    val survivingClusters = clusters.indices.filter(alive).map(clusters)

    val withEarliestBegin = survivingClusters.zipWithIndex.map { case (c, idx) =>
      val earliestBegin =
        if (c.turnIndices.nonEmpty) c.turnIndices.map(turns(_).beginMs).min else Int.MaxValue
      (c, idx, earliestBegin)
    }
    withEarliestBegin.sortBy(t => (t._3, t._2)).foreach { case (c, _, _) =>
      if (c.label == null) {
        c.label = f"SPEAKER_$nextLabelIndex%02d"
        nextLabelIndex += 1
      }
    }

    if (gallery.nonEmpty) {
      survivingClusters.foreach { c =>
        val best = gallery.toSeq
          .map { case (name, ref) => (name, cosineDistance(c.centroid, ref)) }
          .sortBy(t => (t._2, t._1))
          .headOption
        best.foreach { case (name, d) =>
          if (d <= galleryAcceptanceDistance) c.label = name
        }
      }
    }

    def confidenceFor(turnIdx: Int, ownCluster: WorkingCluster): Double = {
      val embedding = turns(turnIdx).embedding
      val distOwn = cosineDistance(embedding, ownCluster.centroid)
      val others = survivingClusters.filter(_ ne ownCluster)
      if (others.isEmpty) 1.0
      else {
        val distNearestOther = others.map(o => cosineDistance(embedding, o.centroid)).min
        val margin = distNearestOther - distOwn
        1.0 / (1.0 + math.exp(
          -margin * 4.0
        ))
      }
    }

    val results = new ArrayBuffer[ClusteredTurn]()
    survivingClusters.foreach { c =>
      c.turnIndices.foreach { idx =>
        results += ClusteredTurn(turns(idx), c.label, confidenceFor(idx, c))
      }
    }

    val resultState = ClusterState(
      centroids =
        survivingClusters.map(c => SpeakerCentroid(c.label, c.centroid, c.memberCount)).toSeq,
      nextLabelIndex = nextLabelIndex)

    val orderById = turns.zipWithIndex.map { case (t, i) => t.id -> i }.toMap
    (results.sortBy(r => orderById(r.turn.id)).toSeq, resultState)
  }

  /** Serializes cluster state to a compact, base64-encoded blob so it can be threaded through a
    * DataFrame column between separate `batchAnnotate` calls — the only mechanism that is correct
    * under arbitrary Spark scheduling (see `SpeakerDiarizer`'s streaming design notes).
    */
  def serializeState(state: ClusterState): String = {
    if (state.centroids.isEmpty) {
      val buffer = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
      buffer.putInt(state.nextLabelIndex)
      buffer.putInt(0)
      Base64.getEncoder.encodeToString(buffer.array())
    } else {
      val dim = state.centroids.head.centroid.length
      val labelBytesPerCentroid = state.centroids.map(_.label.getBytes("UTF-8").length)
      val centroidBytes =
        labelBytesPerCentroid.map(labelLen => 4 + labelLen + 4 + (dim * 4)).sum
      val header = 4 + 4 + 4
      val buffer =
        ByteBuffer.allocate(header + centroidBytes).order(ByteOrder.LITTLE_ENDIAN)
      buffer.putInt(state.nextLabelIndex)
      buffer.putInt(state.centroids.length)
      buffer.putInt(dim)
      state.centroids.foreach { c =>
        val labelBytes = c.label.getBytes("UTF-8")
        buffer.putInt(labelBytes.length)
        buffer.put(labelBytes)
        buffer.putInt(c.memberCount)
        c.centroid.foreach(buffer.putFloat)
      }
      Base64.getEncoder.encodeToString(buffer.array())
    }
  }

  /** Inverse of `serializeState`. */
  def deserializeState(blob: String): ClusterState = {
    val bytes = Base64.getDecoder.decode(blob)
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val nextLabelIndex = buffer.getInt()
    val centroidCount = buffer.getInt()
    if (centroidCount == 0) {
      ClusterState(Seq.empty, nextLabelIndex)
    } else {
      val dim = buffer.getInt()
      val centroids = (0 until centroidCount).map { _ =>
        val labelLen = buffer.getInt()
        val labelBytes = new Array[Byte](labelLen)
        buffer.get(labelBytes)
        val label = new String(labelBytes, "UTF-8")
        val memberCount = buffer.getInt()
        val centroid = Array.fill(dim)(buffer.getFloat())
        SpeakerCentroid(label, centroid, memberCount)
      }
      ClusterState(centroids, nextLabelIndex)
    }
  }
}
