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

// Both dependency-free, deterministic pure-Scala utilities behind SpeakerDiarizer (clustering and
// RTTM interop), mirroring com.johnsnowlabs.ml.ai.util.Generation.Logit.LogitProcess's own
// LogitProcessorTest.scala - one file per small-utility package, not one file per class.
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class DiarizationUtilTestSpec extends AnyFlatSpec {

  // Two well-separated 4-d "speakers" plus noise, so cosine distance cleanly tells them apart
  // regardless of clustering internals - these are not real speaker embeddings, just vectors
  // with the separation properties the algorithm is asserted against.
  private val speakerA = Array(1.0f, 0.0f, 0.0f, 0.0f)
  private val speakerB = Array(0.0f, 1.0f, 0.0f, 0.0f)

  private def jitter(base: Array[Float], seed: Int): Array[Float] =
    base.map(_ + 0.01f * ((seed % 5) - 2))

  private def turnsFromTwoSpeakers(): Seq[SpeakerTurn] = Seq(
    SpeakerTurn("t0", 0, 1000, jitter(speakerA, 0)),
    SpeakerTurn("t1", 1000, 2000, jitter(speakerB, 1)),
    SpeakerTurn("t2", 2000, 3000, jitter(speakerA, 2)),
    SpeakerTurn("t3", 3000, 4000, jitter(speakerB, 3)))

  "SpeakerClustering.cluster" should "be deterministic across repeated runs on identical input" taggedAs FastTest in {
    val turns = turnsFromTwoSpeakers()
    val (resultA, _) = SpeakerClustering.cluster(turns, threshold = 0.3)
    val (resultB, _) = SpeakerClustering.cluster(turns, threshold = 0.3)
    assert(resultA.map(_.speakerLabel) == resultB.map(_.speakerLabel))
  }

  it should "separate two well-separated speakers into two clusters" taggedAs FastTest in {
    val turns = turnsFromTwoSpeakers()
    val (result, _) = SpeakerClustering.cluster(turns, threshold = 0.3)
    val labels = result.map(_.speakerLabel).distinct
    assert(labels.length == 2)
    assert(
      result.find(_.turn.id == "t0").get.speakerLabel ==
        result.find(_.turn.id == "t2").get.speakerLabel)
    assert(
      result.find(_.turn.id == "t1").get.speakerLabel ==
        result.find(_.turn.id == "t3").get.speakerLabel)
    assert(
      result.find(_.turn.id == "t0").get.speakerLabel !=
        result.find(_.turn.id == "t1").get.speakerLabel)
  }

  it should "respect an explicit numSpeakers, overriding the threshold" taggedAs FastTest in {
    val turns = turnsFromTwoSpeakers()
    val (result, _) = SpeakerClustering.cluster(turns, numSpeakers = Some(1), threshold = 0.01)
    assert(result.map(_.speakerLabel).distinct.length == 1)
  }

  it should "force-split a single natural cluster up to numSpeakers when forceExactCount finds fewer natural clusters than requested" taggedAs FastTest in {
    // All four turns are tight jittered copies of the SAME speaker - one natural cluster - but
    // forceExactCount=true (the default, correct for a single complete non-streaming call) must
    // still force exactly 2 distinct labels per this method's own documented contract.
    val turns = Seq(
      SpeakerTurn("t0", 0, 1000, jitter(speakerA, 0)),
      SpeakerTurn("t1", 1000, 2000, jitter(speakerA, 1)),
      SpeakerTurn("t2", 2000, 3000, jitter(speakerA, 2)),
      SpeakerTurn("t3", 3000, 4000, jitter(speakerA, 3)))
    val (result, state) = SpeakerClustering.cluster(turns, numSpeakers = Some(2), threshold = 0.3)
    assert(
      result.map(_.speakerLabel).distinct.length == 2,
      s"expected exactly 2 speakers forced via splitting, got ${result.map(_.speakerLabel).distinct}")
    assert(state.centroids.length == 2)
  }

  it should "not fabricate more clusters than there are turns available to split" taggedAs FastTest in {
    // A single turn can never honestly become 2 clusters - splitStep must give up gracefully
    // (settle for fewer than requested) instead of erroring or duplicating a turn across labels.
    val turns = Seq(SpeakerTurn("t0", 0, 1000, speakerA))
    val (result, _) = SpeakerClustering.cluster(turns, numSpeakers = Some(3), threshold = 0.3)
    assert(result.length == 1)
    assert(result.map(_.speakerLabel).distinct.length == 1)
  }

  it should "NOT split when forceExactCount is false, even with fewer natural clusters than numSpeakers" taggedAs FastTest in {
    // Mirrors an in-progress streaming call: forcing the count up before every real speaker has
    // spoken would manufacture a phantom speaker out of noise - only the merge-down cap applies.
    val turns = Seq(
      SpeakerTurn("t0", 0, 1000, jitter(speakerA, 0)),
      SpeakerTurn("t1", 1000, 2000, jitter(speakerA, 1)))
    val (result, _) =
      SpeakerClustering.cluster(
        turns,
        numSpeakers = Some(2),
        threshold = 0.3,
        forceExactCount = false)
    assert(result.map(_.speakerLabel).distinct.length == 1)
  }

  it should "never produce fewer clusters than minSpeakers" taggedAs FastTest in {
    val turns = turnsFromTwoSpeakers()
    // threshold=1.0 would normally merge everything into one cluster; minSpeakers should stop it
    val (result, _) =
      SpeakerClustering.cluster(turns, threshold = 1.0, minSpeakers = 2, maxSpeakers = 20)
    assert(result.map(_.speakerLabel).distinct.length >= 2)
  }

  it should "never produce more clusters than maxSpeakers" taggedAs FastTest in {
    val manyTurns = (0 until 10).map { i =>
      SpeakerTurn(s"t$i", i * 1000, i * 1000 + 500, Array.fill(4)(scala.util.Random.nextFloat()))
    }
    val (result, _) =
      SpeakerClustering.cluster(manyTurns, threshold = 0.0, minSpeakers = 1, maxSpeakers = 3)
    assert(result.map(_.speakerLabel).distinct.length <= 3)
  }

  it should "assign a single cluster for single-speaker audio, not force a minimum of two" taggedAs FastTest in {
    val turns = Seq(
      SpeakerTurn("t0", 0, 1000, jitter(speakerA, 0)),
      SpeakerTurn("t1", 1000, 2000, jitter(speakerA, 1)),
      SpeakerTurn("t2", 2000, 3000, jitter(speakerA, 2)))
    val (result, _) = SpeakerClustering.cluster(turns, threshold = 0.3, minSpeakers = 1)
    assert(result.map(_.speakerLabel).distinct.length == 1)
  }

  it should "return no output for empty input rather than erroring" taggedAs FastTest in {
    val (result, state) = SpeakerClustering.cluster(Seq.empty)
    assert(result.isEmpty)
    assert(state.centroids.isEmpty)
  }

  it should "produce higher confidence for turns farther from the decision boundary" taggedAs FastTest in {
    val turns = turnsFromTwoSpeakers()
    val (result, _) = SpeakerClustering.cluster(turns, threshold = 0.3)
    result.foreach(r => assert(r.confidence > 0.5))
  }

  it should "rename a cluster to a gallery entry within the acceptance distance" taggedAs FastTest in {
    val turns = turnsFromTwoSpeakers()
    val gallery = Map("Alice" -> speakerA)
    val (result, _) =
      SpeakerClustering.cluster(turns, threshold = 0.3, gallery = gallery)
    val aliceTurns = result.filter(r => r.turn.id == "t0" || r.turn.id == "t2")
    assert(aliceTurns.forall(_.speakerLabel == "Alice"))
    val otherTurns = result.filter(r => r.turn.id == "t1" || r.turn.id == "t3")
    assert(otherTurns.forall(_.speakerLabel != "Alice"))
  }

  it should "carry cluster identity across chunks via priorState" taggedAs FastTest in {
    val (_, stateAfterChunk1) =
      SpeakerClustering.cluster(Seq(SpeakerTurn("c1t0", 0, 1000, speakerA)), threshold = 0.3)
    val labelFromChunk1 = stateAfterChunk1.centroids.head.label

    val (resultChunk2, _) = SpeakerClustering.cluster(
      Seq(SpeakerTurn("c2t0", 5000, 6000, jitter(speakerA, 1))),
      threshold = 0.3,
      priorState = stateAfterChunk1)

    assert(resultChunk2.head.speakerLabel == labelFromChunk1)
  }

  "SpeakerClustering.serializeState / deserializeState" should "round-trip cluster state" taggedAs FastTest in {
    val turns = turnsFromTwoSpeakers()
    val (_, state) = SpeakerClustering.cluster(turns, threshold = 0.3)

    val blob = SpeakerClustering.serializeState(state)
    val restored = SpeakerClustering.deserializeState(blob)

    assert(restored.nextLabelIndex == state.nextLabelIndex)
    assert(restored.centroids.length == state.centroids.length)
    restored.centroids.zip(state.centroids).foreach { case (r, s) =>
      assert(r.label == s.label)
      assert(r.memberCount == s.memberCount)
      assert(r.centroid.zip(s.centroid).forall { case (a, b) => math.abs(a - b) < 1e-6 })
    }
  }

  it should "round-trip empty state" taggedAs FastTest in {
    val blob = SpeakerClustering.serializeState(ClusterState.empty)
    val restored = SpeakerClustering.deserializeState(blob)
    assert(restored.centroids.isEmpty)
    assert(restored.nextLabelIndex == 0)
  }

  "SpeakerClustering.cosineDistance" should "be 0 for identical vectors and >0 for orthogonal ones" taggedAs FastTest in {
    assert(SpeakerClustering.cosineDistance(speakerA, speakerA) < 1e-9)
    assert(SpeakerClustering.cosineDistance(speakerA, speakerB) > 0.9)
  }

  // A very long meeting/recording can produce hundreds of turns even with normal-length speech,
  // since every detected turn (not every audio second) becomes one clustering input - this
  // exercises the O(n^2) incremental-distance-cache rewrite (previously O(n^3) via a full
  // distance rescan on every merge) at a scale where that difference actually matters, and
  // guards against the specific bug that rewrite introduced and fixed mid-session (dead/merged
  // cluster slots leaking into label assignment and final output).
  "SpeakerClustering.cluster at scale" should "handle hundreds of turns from many synthetic speakers without slowing down or misbehaving" taggedAs FastTest in {
    val rng = new scala.util.Random(1234)
    val numSpeakers = 15
    val turnsPerSpeaker = 40
    val dim = 32

    // One well-separated base vector per synthetic speaker (random directions in a fairly high
    // dimension are overwhelmingly likely to be far apart under cosine distance), then several
    // turns per speaker as small jittered copies of their own base - mirrors the real turn/
    // embedding relationship (many turns per real speaker, turns from the same speaker close
    // together, turns from different speakers far apart) without needing real audio or models.
    val speakerBases =
      Array.fill(numSpeakers)(Array.fill(dim)(rng.nextFloat() * 2 - 1))
    val turns = (0 until numSpeakers).flatMap { speakerIdx =>
      (0 until turnsPerSpeaker).map { t =>
        val embedding = speakerBases(speakerIdx).map(_ + (rng.nextFloat() * 0.02f - 0.01f))
        SpeakerTurn(
          id = s"s${speakerIdx}_t$t",
          beginMs = (speakerIdx * turnsPerSpeaker + t) * 1000,
          endMs = (speakerIdx * turnsPerSpeaker + t) * 1000 + 900,
          embedding = embedding)
      }
    }
    assert(turns.length == numSpeakers * turnsPerSpeaker)

    val start = System.nanoTime()
    val (clustered, state) = SpeakerClustering.cluster(turns, threshold = 0.3)
    val elapsedMs = (System.nanoTime() - start) / 1e6

    assert(clustered.length == turns.length, "every input turn must produce exactly one output")
    val distinctLabels = clustered.map(_.speakerLabel).distinct
    assert(
      distinctLabels.length == numSpeakers,
      s"expected exactly $numSpeakers recovered clusters from $numSpeakers well-separated " +
        s"synthetic speakers, got ${distinctLabels.length}: $distinctLabels")
    assert(
      state.centroids.length == numSpeakers,
      "returned state must carry exactly one centroid per real cluster, not a stray leftover " +
        "from a merged-away slot")
    // Every turn from a given synthetic speaker must land in the same output cluster as every
    // other turn from that speaker - not just "some number of clusters", but the RIGHT grouping.
    val labelByTurnId = clustered.map(ct => ct.turn.id -> ct.speakerLabel).toMap
    (0 until numSpeakers).foreach { speakerIdx =>
      val labelsForThisSpeaker =
        (0 until turnsPerSpeaker).map(t => labelByTurnId(s"s${speakerIdx}_t$t")).distinct
      assert(
        labelsForThisSpeaker.length == 1,
        s"all turns from synthetic speaker $speakerIdx should share one label, got $labelsForThisSpeaker")
    }
    println(s"[cluster at scale] ${turns.length} turns, $numSpeakers speakers, ${elapsedMs}ms")
    assert(
      elapsedMs < 10000,
      s"clustering ${turns.length} turns took ${elapsedMs}ms - unexpectedly slow for O(n^2)")
  }

  // ==================== RTTMExporter ====================

  private val rttmAnnotations = Seq(
    Annotation(AnnotatorType.SPEAKER, 0, 3480, "", Map("speaker" -> "SPEAKER_00")),
    Annotation(AnnotatorType.SPEAKER, 3200, 7850, "", Map("speaker" -> "SPEAKER_01")),
    Annotation(AnnotatorType.SPEAKER, 7850, 12010, "", Map("speaker" -> "SPEAKER_00")))

  "RTTMExporter.toRTTM" should "emit one SPEAKER line per annotation" taggedAs FastTest in {
    val rttm = RTTMExporter.toRTTM(rttmAnnotations, "meeting_001")
    val lines = rttm.linesIterator.toSeq
    assert(lines.length == 3)
    assert(lines.forall(_.startsWith("SPEAKER meeting_001 1 ")))
  }

  "RTTMExporter.fromRTTM" should "parse what toRTTM produced, round-tripping begin/end/speaker" taggedAs FastTest in {
    val rttm = RTTMExporter.toRTTM(rttmAnnotations, "meeting_001")
    val parsed = RTTMExporter.fromRTTM(rttm)

    assert(parsed.length == rttmAnnotations.length)
    parsed.zip(rttmAnnotations).foreach { case (segment, annotation) =>
      assert(segment.uri == "meeting_001")
      assert(segment.speaker == annotation.metadata("speaker"))
      assert(math.abs(segment.tbegSeconds - annotation.begin / 1000.0) < 1e-3)
      assert(math.abs(segment.tdurSeconds - (annotation.end - annotation.begin) / 1000.0) < 1e-3)
    }
  }

  it should "skip blank lines and ;; comments rather than erroring" taggedAs FastTest in {
    val rttm = ";; this is a comment\n" + RTTMExporter.toRTTM(rttmAnnotations, "u") + "\n\n"
    val parsed = RTTMExporter.fromRTTM(rttm)
    assert(parsed.length == rttmAnnotations.length)
  }

  it should "skip malformed SPEAKER lines rather than throwing" taggedAs FastTest in {
    val malformed = "SPEAKER u 1 not_a_number 1.0 <NA> <NA> SPEAKER_00 <NA> <NA>"
    assert(RTTMExporter.fromRTTM(malformed).isEmpty)
  }

  it should "return empty for empty input" taggedAs FastTest in {
    assert(RTTMExporter.toRTTM(Seq.empty, "u") == "")
    assert(RTTMExporter.fromRTTM("").isEmpty)
  }
}
