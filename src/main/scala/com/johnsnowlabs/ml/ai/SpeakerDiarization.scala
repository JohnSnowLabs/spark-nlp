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

package com.johnsnowlabs.ml.ai

import ai.onnxruntime.OnnxTensor
import com.johnsnowlabs.ml.ai.util.Diarization._
import com.johnsnowlabs.ml.onnx.OnnxSession
import com.johnsnowlabs.ml.onnx.OnnxWrapper
import com.johnsnowlabs.ml.util.LinAlg
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.KaldiFbank
import com.johnsnowlabs.nlp.{Annotation, AnnotationAudio, AnnotatorType}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

/** Options controlling one `SpeakerDiarization.diarize` call. Grouped into its own case class
  * (rather than a long parameter list) since most fields mirror `SpeakerDiarizer`'s Spark params
  * more or less 1:1 — see that class for the user-facing meaning of each.
  */
case class DiarizationOptions(
    windowDurationSeconds: Float = 10.0f,
    stepDurationSeconds: Float = 1.0f,
    onsetThreshold: Float = 0.5f,
    offsetThreshold: Float = 0.5f,
    minDurationOnSeconds: Float = 0.0f,
    minDurationOffSeconds: Float = 0.0f,
    minSegmentDurationSeconds: Float = 0.5f,
    numSpeakers: Option[Int] = None,
    minSpeakers: Int = 1,
    maxSpeakers: Int = 20,
    clusteringThreshold: Double = 0.7,
    maxChunkDurationSeconds: Float = 300.0f,
    channelMode: String = "mono",
    transcribe: Boolean = true,
    // ASR generation params (used only when transcribe=true) - threaded from SpeakerDiarizer's
    // inherited HasGeneratorProperties getters rather than hardcoded, so those setters actually
    // do something (an earlier version hardcoded these, making every one of those setters a
    // silent no-op - confirmed by testing every parameter, not assumed).
    asrMaxOutputLength: Int = 448,
    asrMinOutputLength: Int = 0,
    asrDoSample: Boolean = false,
    asrBeamSize: Int = 1,
    asrNumReturnSequences: Int = 1,
    asrTemperature: Double = 1.0,
    asrTopK: Int = 1,
    asrTopP: Double = 1.0,
    asrRepetitionPenalty: Double = 1.0,
    asrNoRepeatNgramSize: Int = 0,
    asrRandomSeed: Option[Long] = None,
    asrTask: Option[String] = None,
    asrLanguage: Option[String] = None,
    speakerGallery: Map[String, Array[Float]] = Map.empty,
    galleryAcceptanceDistance: Double = 0.25,
    priorState: ClusterState = ClusterState.empty,
    // Was a bare literal `0.3` inside segmentAndEmbed with no way to reach it - real recordings
    // vary a lot in how the segmentation model's overlap classes behave, so this needed to be
    // tunable rather than baked in.
    overlapDetectionThreshold: Double = 0.3,
    // Real audio context pulled from each neighboring chunk so a turn straddling a chunk boundary
    // is captured whole by at least one chunk's padded slice, instead of being cut into two turns
    // with no way to recombine them - see diarizeSingleClip. Also reused as-is for
    // SpeakerDiarizer's `streamingContextSeconds` param, which used to be threaded nowhere at all.
    chunkOverlapSeconds: Float = 2.0f,
    // A turn's own Annotation.embeddings field used to be populated unconditionally, which is a
    // privacy inconsistency next to speakerGallery's explicit opt-in gating (persistSpeakerGallery)
    // for the exact same kind of biometric data - this makes it opt-in too.
    persistEmbeddings: Boolean = false,
    // true = a single, complete, non-streaming call: safe to force the cluster count down to (or
    // up to, via minSpeakers-style splitting) exactly numSpeakers, since every speaker in the
    // recording has already been seen. false = one call in an ongoing streaming session (or,
    // historically, one intermediate chunk of one call - no longer applicable now that chunking
    // is purely an internal segmentation/embedding compute detail, see diarizeSingleClip): forcing
    // the exact count before every real speaker has spoken manufactures a phantom cluster out of
    // noise that then persists across calls via priorState. SpeakerDiarizer sets this to
    // `!streamingMode`.
    forceExactSpeakerCount: Boolean = true,
    // WeSpeaker's embedding model was validated on utterance-length crops; an extremely long turn
    // (e.g. one speaker monologuing for minutes with no detected pause) is cropped to this many
    // seconds before embedding rather than fed in whole, which the model was never validated on.
    maxEmbeddingClipSeconds: Float = 30.0f,
    // Whisper's encoder takes a fixed-size, exactly-30-second input window by construction
    // (3000 mel frames) regardless of how much audio is actually handed to it - a turn longer than
    // this is truncated here explicitly, with a logged warning, rather than relying on whatever
    // silent truncation/error the model does internally.
    maxAsrClipSeconds: Float = 30.0f)

/** Orchestrates the three bundled sub-models behind `SpeakerDiarizer` — segmentation, speaker
  * embedding, and (optionally) Whisper ASR — plus the pure-Scala clustering, overlap surfacing,
  * and output assembly around them. No Spark `Param`s live here; this class only knows how to
  * turn raw audio into labeled `Annotation`s, mirroring the `Whisper` / `WhisperForCTC` split.
  *
  * '''Segmentation model I/O''' — verified against `onnx-community/pyannote-segmentation-3.0`
  * (MIT, a non-gated ONNX export of `pyannote/segmentation-3.0`): input `"input_values"` shape
  * `[batch, 1, numSamples]` at `samplingRate` Hz; output `"logits"` shape `[batch, numFrames, 7]`
  * — raw logits (softmax applied in `runSegmentationModel`) over `[NO_SPEAKER, SPEAKER_1,
  * SPEAKER_2, SPEAKER_3, SPEAKERS_1_AND_2, SPEAKERS_1_AND_3, SPEAKERS_2_AND_3]` per that model's
  * `config.json`.
  *
  * '''Embedding model I/O''' — verified against `Wespeaker/wespeaker-voxceleb-resnet34-LM`
  * (CC-BY-4.0, the WeSpeaker project's own official ONNX export): input `"feats"` shape `[batch,
  * numFrames, 80]` — 80-dim Kaldi-style log-mel filterbank features, '''not''' raw waveform (see
  * `KaldiFbank`) — output `"embs"` shape `[batch, 256]`.
  *
  * Both were downloaded and run against real multi-speaker audio (two distinct LibriSpeech
  * speakers, three turns) during development: segmentation correctly recovered all three turn
  * boundaries and the embedding model gave 0.86 cosine similarity between the two same-speaker
  * turns versus ~0.12 between different speakers — see the conversion notebook for the full
  * validation run.
  */
private[johnsnowlabs] class SpeakerDiarization(
    val segmentationWrapper: OnnxWrapper,
    val embeddingWrapper: OnnxWrapper,
    val whisper: Option[Whisper],
    samplingRate: Int)
    extends Serializable {

  private val logger = LoggerFactory.getLogger(this.getClass.getName)
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  /** Converts a sample index to milliseconds, failing loudly rather than silently wrapping if it
    * overflows `Int` (`Annotation.begin`/`end` are `Int`-typed milliseconds, so a recording
    * longer than ~24.8 days has no representable timestamp for its later samples - better to
    * reject that input outright than to hand back a wrapped, nonsensical negative or small
    * timestamp).
    */
  private def toMillis(sampleIndex: Long): Int = {
    val ms = math.round(sampleIndex.toDouble / samplingRate * 1000)
    if (ms > Int.MaxValue || ms < 0) {
      throw new IllegalArgumentException(
        s"Audio position $sampleIndex samples (~${ms}ms) exceeds the maximum timestamp " +
          "SpeakerDiarizer can represent (Int.MaxValue milliseconds, ~24.8 days), since " +
          "Annotation.begin/end are Int-typed milliseconds. Split this recording into shorter " +
          "inputs before diarizing.")
    }
    ms.toInt
  }

  /** Like `toMillis`, but adds a sample offset (within some chunk/slice) to an already-known
    * absolute millisecond base, instead of converting an absolute sample index directly - used
    * wherever a turn's position is computed relative to a chunk's own start.
    */
  private def offsetMillis(baseMs: Int, sampleOffsetWithinChunk: Long): Int = {
    val ms = baseMs.toLong + math.round(sampleOffsetWithinChunk.toDouble / samplingRate * 1000)
    if (ms > Int.MaxValue || ms < 0) {
      throw new IllegalArgumentException(
        s"Turn timestamp ${ms}ms exceeds the maximum SpeakerDiarizer can represent " +
          "(Int.MaxValue milliseconds, ~24.8 days), since Annotation.begin/end are Int-typed " +
          "milliseconds. Split this recording into shorter inputs before diarizing.")
    }
    ms.toInt
  }

  /** Crops `clip` to at most `maxSeconds`, taken from its start, logging once per call if it had
    * to. Used before both embedding (WeSpeaker was validated on utterance-length audio) and ASR
    * (Whisper's encoder takes a fixed 30s input by construction) to keep an unusually long turn
    * from being fed to either model far outside what it was built for.
    */
  private def cropToMaxSeconds(clip: Array[Float], maxSeconds: Float): Array[Float] = {
    val maxSamples = (maxSeconds * samplingRate).toInt
    if (maxSamples > 0 && clip.length > maxSamples) {
      logger.warn(
        f"Cropping a ${clip.length.toDouble / samplingRate}%.1fs turn down to " +
          f"$maxSeconds%.1fs before running inference on it.")
      clip.take(maxSamples)
    } else clip
  }

  /** Runs the full pipeline over one batch of audio rows and returns one `Seq[Annotation]` per
    * input row (matching `HasBatchedAnnotateAudio`'s per-row contract) plus the resulting cluster
    * state for the last row processed, for streaming continuation.
    */
  def diarize(
      batchAudio: Seq[AnnotationAudio],
      options: DiarizationOptions): (Seq[Seq[Annotation]], ClusterState) = {

    if (options.channelMode == "stereo") return diarizeStereo(batchAudio)

    var runningState = options.priorState
    val perRowAnnotations = batchAudio.map { annotationAudio =>
      if (annotationAudio.result.isEmpty) Seq.empty[Annotation]
      else {
        val (annotations, newState) =
          diarizeSingleClip(annotationAudio, options, runningState)
        runningState = newState
        annotations
      }
    }
    (perRowAnnotations, runningState)
  }

  /** `channelMode = "stereo"`: each input row is expected to carry two `AnnotationAudio` (left,
    * right channel, via `setInputCols(Array(leftCol, rightCol))`), and each channel is treated as
    * exactly one speaker for its whole duration — no segmentation, embedding, or clustering at
    * all, since the recording already tells us who is who. Cheaper and more accurate than
    * rediscovering that with the ML pipeline when it's available.
    */
  private def diarizeStereo(
      batchAudio: Seq[AnnotationAudio]): (Seq[Seq[Annotation]], ClusterState) = {
    val channelLabels = Array("SPEAKER_00", "SPEAKER_01")
    val grouped = batchAudio.grouped(2).toSeq
    val perRow = grouped.map { channels =>
      channels.zipWithIndex.map { case (audio, channelIdx) =>
        val durationMs = toMillis(audio.result.length.toLong)
        new Annotation(
          annotatorType = AnnotatorType.SPEAKER,
          begin = 0,
          end = durationMs,
          result = "",
          metadata = audio.metadata ++ Map(
            "speaker" -> channelLabels(math.min(channelIdx, channelLabels.length - 1)),
            "channel" -> channelIdx.toString,
            "confidence" -> "1.0000"))
      }
    }
    (perRow, ClusterState.empty)
  }

  /** The result of segmenting+embedding a whole clip, chunk by chunk, with boundary turns already
    * stitched back together — see `collectStitchedTurns`.
    */
  private case class CollectedTurns(
      turns: Seq[SpeakerTurn],
      overlapById: Map[String, Boolean],
      shortSegmentById: Map[String, Boolean])

  private def diarizeSingleClip(
      annotationAudio: AnnotationAudio,
      options: DiarizationOptions,
      priorState: ClusterState): (Seq[Annotation], ClusterState) = {

    val samples = annotationAudio.result
    val collected = collectStitchedTurns(samples, options)

    // Clustering runs once, over every turn in the whole clip, regardless of how many internal
    // chunks it took to segment+embed them - chunking here is purely a compute/memory-bounding
    // detail (see collectStitchedTurns), not a clustering boundary, so it can no longer manufacture
    // a premature/phantom speaker cluster the way per-chunk clustering used to. `priorState` (an
    // actual previous streaming call, or the empty state for a one-shot call) and
    // `forceExactSpeakerCount` (false for an in-progress streaming session, true for a complete
    // single call - see DiarizationOptions' scaladoc) are what still make cross-call speaker
    // identity correct.
    val (clustered, state) = SpeakerClustering.cluster(
      collected.turns,
      numSpeakers = options.numSpeakers,
      minSpeakers = options.minSpeakers,
      maxSpeakers = options.maxSpeakers,
      threshold = options.clusteringThreshold,
      priorState = priorState,
      gallery = options.speakerGallery,
      galleryAcceptanceDistance = options.galleryAcceptanceDistance,
      forceExactCount = options.forceExactSpeakerCount)

    val annotations = clustered.map { ct =>
      val (transcript, transcriptionError) =
        if (options.transcribe) transcribeTurn(samples, ct.turn, options) else ("", None)
      // A short-duration crop still clusters, but its embedding is less reliable than one drawn
      // from a full-length turn - discount the reported confidence rather than presenting it with
      // the same weight as a well-formed turn (see minSegmentDurationSeconds/shortSegmentById).
      val isShort = collected.shortSegmentById.getOrElse(ct.turn.id, false)
      val reportedConfidence = if (isShort) ct.confidence * 0.5 else ct.confidence
      val baseMetadata = annotationAudio.metadata ++ Map(
        "speaker" -> ct.speakerLabel,
        "confidence" -> f"$reportedConfidence%.4f",
        "overlap" -> collected.overlapById.getOrElse(ct.turn.id, false).toString,
        "channel" -> "0")
      // Distinguishes a genuine ASR failure from genuine silence: an empty result with no
      // "transcriptionError" key means the model really did decode nothing (or transcribe=false);
      // an empty result with this key present means inference on this turn threw, and the empty
      // string is a fallback, not a claim about what was said.
      val metadata = transcriptionError
        .map(err => baseMetadata + ("transcriptionError" -> err))
        .getOrElse(baseMetadata)
      new Annotation(
        annotatorType = AnnotatorType.SPEAKER,
        begin = ct.turn.beginMs,
        end = ct.turn.endMs,
        result = transcript,
        metadata = metadata,
        // Exposes the turn's raw voice embedding on the free Annotation.embeddings field, so a
        // caller can do their own similarity matching (e.g. against a voiceprint enrolled after
        // the fact) without re-running inference. Gated behind persistEmbeddings (default false)
        // for the same reason speakerGallery persistence is gated behind persistSpeakerGallery:
        // a voice embedding is biometric data, and shouldn't leave the annotator by default.
        embeddings = if (options.persistEmbeddings) ct.turn.embedding else Array.emptyFloatArray)
    }

    (annotations, state)
  }

  /** Splits `samples` into `maxChunkDurationSeconds`-sized "core" ranges purely to bound the
    * memory/compute of segmentation+embedding on very long recordings, then stitches back
    * together any turn that straddles a core boundary so the chunking is invisible in the
    * returned turns.
    *
    * Each core range's actual audio slice is padded by `chunkOverlapSeconds` on each side
    * (clipped to the clip's bounds), so a turn crossing a boundary is captured '''in full''' by
    * at least one neighboring chunk's padded slice — the previous implementation split chunks
    * with zero overlap (`samples.grouped(chunkSize)`), so any turn straddling a boundary was
    * simply cut into two independent turns with nothing to recombine them.
    *
    * For each chunk, turns are classified against that chunk's own core range: fully inside it
    * ("interior", kept as-is), starting before it ("leading", only possible for a non-first
    * chunk, since it means the turn was already partly seen by the previous chunk's trailing
    * padding), or ending after it ("trailing", only possible for a non-last chunk, held as
    * "pending" for the next chunk to try to complete). A pending trailing turn is stitched with a
    * next chunk's leading turn when they are time-adjacent (within one `chunkOverlapSeconds` of
    * each other, allowing for the small gap segmentation can leave right at a cut) '''and'''
    * cosine-close under `clusteringThreshold` — i.e., actually the same voice — in which case
    * they're replaced by one turn spanning both, re-embedded from the original full-resolution
    * `samples` (not either chunk's local slice, which would only ever contain half the true
    * turn). An unmatched pending or leading turn is kept independently: it genuinely ended (or
    * started) right at the boundary by coincidence, not because of it.
    */
  private def collectStitchedTurns(
      samples: Array[Float],
      options: DiarizationOptions): CollectedTurns = {

    if (samples.isEmpty) return CollectedTurns(Seq.empty, Map.empty, Map.empty)

    val chunkSizeSamples = math.max(1, (options.maxChunkDurationSeconds * samplingRate).toInt)
    val contextSamples = math.max(
      0,
      math.min(chunkSizeSamples / 2, (options.chunkOverlapSeconds * samplingRate).toInt))
    val adjacencyGapMs = math.round(options.chunkOverlapSeconds * 1000)

    val coreRanges = new ArrayBuffer[(Int, Int)]()
    var coreCursor = 0
    while (coreCursor < samples.length) {
      val coreEnd = math.min(coreCursor + chunkSizeSamples, samples.length)
      coreRanges += ((coreCursor, coreEnd))
      coreCursor = coreEnd
    }

    val overlapById = scala.collection.mutable.Map.empty[String, Boolean]
    val shortSegmentById = scala.collection.mutable.Map.empty[String, Boolean]
    val resolvedTurns = new ArrayBuffer[SpeakerTurn]()
    var pendingTrailing: Seq[SpeakerTurn] = Seq.empty

    coreRanges.zipWithIndex.foreach { case ((coreStartSample, coreEndSample), idx) =>
      val isFirst = idx == 0
      val isLast = idx == coreRanges.length - 1
      val sliceStart = math.max(0, coreStartSample - contextSamples)
      val sliceEnd = math.min(samples.length, coreEndSample + contextSamples)
      val slice = samples.slice(sliceStart, sliceEnd)

      val (turns, overlapFlags, shortFlags) =
        segmentAndEmbed(slice, toMillis(sliceStart.toLong), options)
      overlapById ++= overlapFlags
      shortSegmentById ++= shortFlags

      val coreStartMs = toMillis(coreStartSample.toLong)
      val coreEndMs = toMillis(coreEndSample.toLong)

      val leading = new ArrayBuffer[SpeakerTurn]()
      val interior = new ArrayBuffer[SpeakerTurn]()
      val trailing = new ArrayBuffer[SpeakerTurn]()
      turns.foreach { t =>
        if (!isFirst && t.beginMs < coreStartMs) leading += t
        else if (!isLast && t.endMs > coreEndMs) trailing += t
        else interior += t
      }

      // Candidates leaving this boundary-resolution step (merged, or left unmatched) still need to
      // be checked against THIS chunk's own coreEnd before being finalized: a turn spanning more
      // than two chunks (only reachable with an unusually small maxChunkDurationSeconds relative
      // to chunkOverlapSeconds/turn length, as in a stress test, but not impossible) needs to stay
      // a pending candidate across every boundary it crosses, not just the first one.
      val resolvedHere = new ArrayBuffer[SpeakerTurn]()
      val usedLeading = scala.collection.mutable.Set.empty[Int]
      val usedPending = scala.collection.mutable.Set.empty[Int]
      pendingTrailing.zipWithIndex.foreach { case (p, pIdx) =>
        var bestMatch: Option[(Int, Double)] = None
        leading.zipWithIndex.foreach { case (l, lIdx) =>
          if (!usedLeading.contains(lIdx)) {
            // A real boundary-split turn can either abut cleanly (a small gap either way, up to
            // one chunkOverlapSeconds) or - when the padding is generous relative to the chunk
            // size - be independently (re)detected by both neighbors as substantially the same,
            // heavily time-overlapping span with slightly different VAD-onset boundaries. Both are
            // "the same turn cut by chunking" cases; a plain signed-gap check only recognizes the
            // first, and misses (double-counts) the second.
            val isAdjacentOrOverlapping =
              l.beginMs <= p.endMs + adjacencyGapMs && p.beginMs <= l.endMs + adjacencyGapMs
            if (isAdjacentOrOverlapping) {
              val d = SpeakerClustering.cosineDistance(p.embedding, l.embedding)
              if (d <= options.clusteringThreshold && bestMatch.forall(_._2 > d))
                bestMatch = Some((lIdx, d))
            }
          }
        }
        bestMatch match {
          case Some((lIdx, _)) =>
            val l = leading(lIdx)
            usedLeading += lIdx
            usedPending += pIdx
            val mergedBeginMs = math.min(p.beginMs, l.beginMs)
            val mergedEndMs = math.max(p.endMs, l.endMs)
            val mergedBeginSample = math.round(mergedBeginMs / 1000.0 * samplingRate).toInt
            val mergedEndSample = math.round(mergedEndMs / 1000.0 * samplingRate).toInt
            val clip = samples.slice(
              math.max(0, mergedBeginSample),
              math.min(samples.length, mergedEndSample))
            val reembedded =
              if (clip.nonEmpty)
                runEmbeddingModel(cropToMaxSeconds(clip, options.maxEmbeddingClipSeconds))
              else Array.emptyFloatArray
            val embedding = if (reembedded.nonEmpty) reembedded else p.embedding
            val mergedId = s"${p.id}+${l.id}"
            overlapById(mergedId) =
              overlapById.getOrElse(p.id, false) || overlapById.getOrElse(l.id, false)
            shortSegmentById(mergedId) =
              (mergedEndMs - mergedBeginMs) / 1000.0 < options.minSegmentDurationSeconds
            resolvedHere += SpeakerTurn(mergedId, mergedBeginMs, mergedEndMs, embedding)
          case None => ()
        }
      }
      // A pending turn with no match genuinely ended right at the previous boundary; a leading
      // turn with no match genuinely starts right at this one - neither was actually cut, so both
      // are kept independently rather than dropped.
      pendingTrailing.zipWithIndex.foreach { case (p, pIdx) =>
        if (!usedPending.contains(pIdx)) resolvedHere += p
      }
      leading.zipWithIndex.foreach { case (l, lIdx) =>
        if (!usedLeading.contains(lIdx)) resolvedHere += l
      }
      resolvedHere.foreach { t =>
        // Still crosses THIS chunk's own trailing boundary (a turn spanning 3+ chunks) - keep it
        // alive as a pending candidate for the next chunk instead of finalizing it here.
        if (!isLast && t.endMs > coreEndMs) trailing += t else resolvedTurns += t
      }
      resolvedTurns ++= interior
      pendingTrailing = trailing.toSeq
    }
    // By construction, the last core range never populates `trailing` (isLast is true for it), so
    // nothing is ever left pending here - flushed anyway as a defensive no-op.
    resolvedTurns ++= pendingTrailing

    CollectedTurns(resolvedTurns.toSeq, overlapById.toMap, shortSegmentById.toMap)
  }

  /** Runs the segmentation model over sliding windows of one chunk and aggregates its per-frame
    * output into a continuous, sample-resolution activity curve before binarizing into turns —
    * deliberately '''not''' matching each window's discrete per-frame argmax class against the
    * previous window's, because pyannote's powerset local-speaker-slot numbering is only
    * consistent '''within''' one window, not across overlapping windows (there is nothing tying
    * "local speaker 1" in one 10s window to "local speaker 1" in the next window shifted by 1s —
    * empirically, the same real speaker flips between slots across windows). An earlier version
    * of this method matched discrete per-window labels directly, which under that permutation
    * ambiguity fragmented long turns into many sub-`minDurationOn` pieces and silently dropped
    * them — confirmed against real segmentation-3.0 output on real multi-speaker audio, not a
    * hypothetical. Aggregating a label-agnostic "is anyone speaking" curve (`1 - P(silence)`)
    * across every window covering a given sample sidesteps the permutation problem entirely:
    * onset/offset hysteresis over that curve is what actually delimits turns.
    *
    * A second aggregated curve, `P(any 2-of-3 overlap class)`, flags turns as `overlap` in the
    * final annotation's metadata (see `diarizeSingleClip`) rather than attempting to emit two
    * separate per-speaker turns for an overlap: this class of model gives one embedding for
    * whatever audio it's fed, so an overlapping crop only ever yields one (unreliable, blended)
    * embedding — real per-speaker separation during overlap needs source separation, out of scope
    * here. Flagging it lets a downstream consumer treat that turn's speaker assignment with
    * appropriate skepticism instead of the annotator quietly asserting false precision.
    *
    * Turns are then embedded via [[runEmbeddingModel]]. Turns shorter than
    * `minSegmentDurationSeconds` are still embedded and returned (per the edge-case requirement
    * to never silently drop short speech), but flagged in the returned map so the caller can
    * discount their confidence — `minDurationOnSeconds` (not `minSegmentDurationSeconds`) is what
    * actually filters pure noise bursts before they reach embedding.
    *
    * @return
    *   the embedded turns, a turn-id -> overlap-detected map, and a turn-id -> below-
    *   minSegmentDurationSeconds map
    */
  private def segmentAndEmbed(
      chunkSamples: Array[Float],
      chunkOffsetMs: Int,
      options: DiarizationOptions)
      : (Seq[SpeakerTurn], Map[String, Boolean], Map[String, Boolean]) = {

    if (chunkSamples.isEmpty) return (Seq.empty, Map.empty, Map.empty)

    val windowSize = (options.windowDurationSeconds * samplingRate).toInt
    val stepSize = math.max(1, (options.stepDurationSeconds * samplingRate).toInt)
    val n = chunkSamples.length
    // The final sliding window always ends exactly at `n` (see below), so whenever `n` isn't an
    // exact multiple of `stepSize` past the previous window start, that last window can be just a
    // handful of samples - the segmentation model's SincNet frontend errors on an input that
    // small (confirmed: `ORT_INVALID_ARGUMENT ... Invalid input shape: {3}` on a real 3-sample
    // window during testing) rather than degrading gracefully. Skipping windows below a sane floor
    // avoids feeding it something outside what it was built for; losing coverage for a sliver
    // under 100ms has no meaningful effect on a turn boundary that's already rounded to whole
    // milliseconds.
    val minWindowSamples = math.max(1, samplingRate / 10)

    val activitySum = new Array[Double](n)
    val overlapSum = new Array[Double](n)
    val coverage = new Array[Int](n)

    var windowStart = 0
    while (windowStart < n) {
      val windowEnd = math.min(windowStart + windowSize, n)
      val window = chunkSamples.slice(windowStart, windowEnd)
      val (speechActivity, overlapActivity) =
        if (window.length >= minWindowSamples) runSegmentationModel(window)
        else (Array.empty[Float], Array.empty[Float])
      val numFrames = speechActivity.length
      if (numFrames > 0) {
        val frameDurationSamples = math.max(1, window.length / numFrames)
        var i = 0
        while (i < numFrames) {
          val frameStart = windowStart + i * frameDurationSamples
          // The last frame's nominal end (frameStart + frameDurationSamples) truncates via the
          // integer division above and so falls short of the window's true end whenever
          // window.length isn't an exact multiple of numFrames - the common case. Snapping the
          // last frame's end to the window's own true end (windowStart + window.length) instead of
          // frameStart + frameDurationSamples absorbs that remainder instead of leaving a few
          // trailing samples with coverage=0 (read as silence by the hysteresis loop below
          // regardless of their true content) whenever no other, earlier overlapping window
          // happens to cover them - guaranteed impossible for the very last window of a chunk,
          // since no later window exists to compensate.
          val frameEnd =
            if (i == numFrames - 1) math.min(windowStart + window.length, n)
            else math.min(frameStart + frameDurationSamples, n)
          var s = frameStart
          while (s < frameEnd) {
            activitySum(s) += speechActivity(i)
            overlapSum(s) += overlapActivity(i)
            coverage(s) += 1
            s += 1
          }
          i += 1
        }
      }
      windowStart += stepSize
    }

    // Sample-level onset/offset hysteresis over the coverage-averaged activity curve.
    val isSpeech = new Array[Boolean](n)
    var active = false
    var i = 0
    while (i < n) {
      val cov = math.max(1, coverage(i))
      val activity = activitySum(i) / cov
      if (!active && activity >= options.onsetThreshold) active = true
      else if (active && activity < options.offsetThreshold) active = false
      isSpeech(i) = active
      i += 1
    }

    val minDurationOnSamples = (options.minDurationOnSeconds * samplingRate).toInt
    val minGapSamples = (options.minDurationOffSeconds * samplingRate).toInt

    val rawTurns = new ArrayBuffer[(Int, Int)]()
    i = 0
    while (i < n) {
      if (isSpeech(i)) {
        val start = i
        while (i < n && isSpeech(i)) i += 1
        rawTurns += ((start, i))
      } else i += 1
    }

    val mergedTurns = new ArrayBuffer[(Int, Int)]()
    rawTurns.foreach { t =>
      if (mergedTurns.nonEmpty && t._1 - mergedTurns.last._2 <= minGapSamples) {
        mergedTurns(mergedTurns.length - 1) = (mergedTurns.last._1, t._2)
      } else {
        mergedTurns += t
      }
    }
    val finalTurns = mergedTurns.filter { case (s, e) => (e - s) >= minDurationOnSamples }

    val turns = new ArrayBuffer[SpeakerTurn]()
    val overlapFlags = scala.collection.mutable.Map.empty[String, Boolean]
    val shortSegmentFlags = scala.collection.mutable.Map.empty[String, Boolean]
    finalTurns.zipWithIndex.foreach { case ((startSample, endSample), idx) =>
      val clip = chunkSamples.slice(startSample, endSample)
      if (clip.nonEmpty) {
        val embedding = runEmbeddingModel(cropToMaxSeconds(clip, options.maxEmbeddingClipSeconds))
        // A turn shorter than one fbank analysis window (25ms) or whose embedding inference
        // failed has no vector to cluster by at all — there is no honest way to still surface it
        // with a speaker assignment, so it's dropped here rather than clustered with a fabricated
        // placeholder embedding. This is a hard floor set by the embedding model's own minimum
        // input requirement, distinct from `minDurationOnSeconds` (which governs when a
        // detected-speech run becomes a turn in the first place) — turns clearing that gate can
        // still be too short to embed, and that's what this guards against.
        if (embedding.nonEmpty) {
          val beginMs = offsetMillis(chunkOffsetMs, startSample)
          val endMs = offsetMillis(chunkOffsetMs, endSample)
          val meanOverlap = {
            var sum = 0.0
            var s = startSample
            while (s < endSample) {
              sum += overlapSum(s) / math.max(1, coverage(s))
              s += 1
            }
            sum / (endSample - startSample)
          }
          val id = s"turn-$chunkOffsetMs-$idx"
          turns += SpeakerTurn(id, beginMs, endMs, embedding)
          overlapFlags(id) = meanOverlap >= options.overlapDetectionThreshold
          // A short crop gives a less reliable voice embedding even when it's long enough to
          // embed at all - flagged here so the final confidence score (computed later, once
          // clustering has run) can be discounted for it, per minSegmentDurationSeconds.
          val durationSeconds = (endSample - startSample).toDouble / samplingRate
          shortSegmentFlags(id) = durationSeconds < options.minSegmentDurationSeconds
        }
      }
    }
    (turns.toSeq, overlapFlags.toMap, shortSegmentFlags.toMap)
  }

  /** Validates an ASR task/language token against the actually-loaded model's vocabulary rather
    * than only the format-level check `SpeakerDiarizer.setAsrTask`/`setAsrLanguage` can do at
    * `set*`-time (the model may not even be loaded yet when those setters run). A token that
    * looks right but isn't in this particular export's vocabulary (e.g. a language the loaded
    * Whisper tier doesn't support) is dropped with a warning instead of being silently passed
    * through to fail (or worse, be silently ignored) deep inside generation.
    */
  private def validateAsrToken(
      token: Option[String],
      kind: String,
      isKnown: String => Boolean): Option[String] =
    token.filter { t =>
      val known = isKnown(t)
      if (!known)
        logger.warn(
          s"asr$kind '$t' is not in the loaded ASR model's vocabulary; ignoring it and letting " +
            "the model use its default instead.")
      known
    }

  /** @return
    *   the transcript text, and — when transcription was attempted but failed — `Some(message)`
    *   distinguishing a genuine inference failure from genuine silence (both of which otherwise
    *   surface as the same empty-string `result`).
    */
  private def transcribeTurn(
      fullClipSamples: Array[Float],
      turn: SpeakerTurn,
      options: DiarizationOptions): (String, Option[String]) = {
    whisper match {
      case None => ("", None)
      case Some(model) =>
        val startSample = math.max(0, (turn.beginMs / 1000.0 * samplingRate).toInt)
        val endSample =
          math.min(fullClipSamples.length, (turn.endMs / 1000.0 * samplingRate).toInt)
        if (endSample <= startSample) ("", None)
        else {
          val crop = cropToMaxSeconds(
            fullClipSamples.slice(startSample, endSample),
            options.maxAsrClipSeconds)
          val cropAnnotation = AnnotationAudio(AnnotatorType.AUDIO, crop, Map.empty)
          try {
            val result = model.generateFromAudio(
              batchAudio = Seq(cropAnnotation),
              batchSize = 1,
              maxOutputLength = options.asrMaxOutputLength,
              minOutputLength = options.asrMinOutputLength,
              doSample = options.asrDoSample,
              beamSize = options.asrBeamSize,
              numReturnSequences = options.asrNumReturnSequences,
              temperature = options.asrTemperature,
              topK = options.asrTopK,
              topP = options.asrTopP,
              repetitionPenalty = options.asrRepetitionPenalty,
              noRepeatNgramSize = options.asrNoRepeatNgramSize,
              randomSeed = options.asrRandomSeed,
              // model.tokenInVocabulary, not the bare model.vocabulary map: real task/language
              // tokens (e.g. <|transcribe|>, <|en|>) live in addedSpecialTokens, not in vocabulary
              // itself - checking the bare map here meant every legitimate asrTask/asrLanguage
              // value was always rejected as "unknown" and silently discarded.
              task = validateAsrToken(options.asrTask, "Task", model.tokenInVocabulary),
              language =
                validateAsrToken(options.asrLanguage, "Language", model.tokenInVocabulary),
              outputTimestamps = false)
            (result.headOption.flatMap(_.headOption).map(_.result.trim).getOrElse(""), None)
          } catch {
            case e: Exception =>
              val message = Option(e.getMessage).getOrElse(e.getClass.getSimpleName)
              logger.warn(s"Failed to transcribe a speaker turn, leaving it blank: $message")
              ("", Some(message))
          }
        }
    }
  }

  /** Runs the segmentation model on one window and returns two per-frame curves derived from its
    * softmax'd powerset class probabilities: `speechActivity` (`1 - P(silence)`, class 0) and
    * `overlapActivity` (sum of the three 2-of-3-concurrent classes 4-6). Both are label-agnostic
    * — they never depend on which of the three local speaker slots was active — which is what
    * makes them safe to aggregate across overlapping windows despite pyannote's local slot
    * numbering not being consistent window-to-window (see `segmentAndEmbed`'s scaladoc).
    *
    * Verified I/O against the real exported model (`onnx-community/pyannote-segmentation-3.0`):
    * input `"input_values"` shape `[batch, channels, numSamples]`, output `"logits"` shape
    * `[batch, numFrames, 7]` (raw logits, softmax applied here) with class order `[NO_SPEAKER,
    * SPEAKER_1, SPEAKER_2, SPEAKER_3, SPEAKERS_1_AND_2, SPEAKERS_1_AND_3, SPEAKERS_2_AND_3]` per
    * that model's `config.json`.
    */
  private def runSegmentationModel(window: Array[Float]): (Array[Float], Array[Float]) = {
    val (session, env) = segmentationWrapper.getSession(onnxSessionOptions)
    val inputTensor = OnnxTensor.createTensor(env, Array(Array(window)))
    try {
      val results = session.run(Map("input_values" -> inputTensor).asJava)
      try {
        val outputTensor = results.get("logits").get().asInstanceOf[OnnxTensor]
        val shape = outputTensor.getInfo.getShape // [1, numFrames, 7]
        val numFrames = shape(1).toInt
        val numClasses = shape(2).toInt
        val flat = outputTensor.getFloatBuffer.array()

        val speechActivity = new Array[Float](numFrames)
        val overlapActivity = new Array[Float](numFrames)
        val logits = new Array[Float](numClasses)
        var f = 0
        while (f < numFrames) {
          val base = f * numClasses
          var c = 0
          while (c < numClasses) {
            logits(c) = flat(base + c)
            c += 1
          }
          val probs = LinAlg.softmax(logits)
          speechActivity(f) = 1.0f - probs(0)
          overlapActivity(f) = if (numClasses >= 7) probs(4) + probs(5) + probs(6) else 0.0f
          f += 1
        }
        (speechActivity, overlapActivity)
      } finally if (results != null) results.close()
    } catch {
      case e: Exception =>
        logger.error(s"Segmentation model inference failed: ${e.getMessage}", e)
        (Array.empty[Float], Array.empty[Float])
    } finally {
      inputTensor.close()
    }
  }

  /** Returns one embedding vector for `clip`.
    *
    * Verified I/O against the real exported model (`Wespeaker/wespeaker-voxceleb-resnet34-LM`):
    * input `"feats"` shape `[batch, numFrames, 80]` — 80-dim Kaldi-style log-mel filterbank
    * features (25ms/10ms, per-utterance mean-normalized), '''not''' raw waveform samples — output
    * `"embs"` shape `[batch, 256]`. Feature extraction is
    * [[KaldiFbank.extractNormalizedFeatures]] (see that class for the exact recipe and its
    * validation against `torchaudio.compliance.kaldi.fbank`).
    */
  private def runEmbeddingModel(clip: Array[Float]): Array[Float] = {
    val features = KaldiFbank.extractNormalizedFeatures(clip, sampleFrequency = samplingRate)
    if (features.isEmpty) return Array.emptyFloatArray

    val (session, env) = embeddingWrapper.getSession(onnxSessionOptions)
    val inputTensor = OnnxTensor.createTensor(env, Array(features))
    try {
      val results = session.run(Map("feats" -> inputTensor).asJava)
      try {
        results.get("embs").get().asInstanceOf[OnnxTensor].getFloatBuffer.array()
      } finally if (results != null) results.close()
    } catch {
      case e: Exception =>
        logger.error(s"Embedding model inference failed: ${e.getMessage}", e)
        Array.empty[Float]
    } finally {
      inputTensor.close()
    }
  }
}
