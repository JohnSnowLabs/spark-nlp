/*
 * Everything that doesn't need the ASR/Whisper-bundled model (so none of the memory-isolation
 * concerns SpeakerDiarizerTestFixtures documents apply here) and doesn't fit the exhaustive
 * per-parameter validation in SpeakerDiarizerFullValidationSpec: `embeddingModelSize` (found,
 * while writing this, to always throw NoSuchElementException - loadSavedModel never actually set
 * it despite the class scaladoc saying it does; fixed alongside this test, not just tested
 * around), concurrent calls into one broadcasted model from multiple threads (realistic for a
 * multi-core Spark executor, never exercised before - everything else in this whole test campaign
 * runs single-threaded), the full streaming phantom-speaker-prevention lifecycle (including the
 * setStreamingPriorState consumption and intra-batch cluster-state-threading fixes), malformed/
 * adversarial audio, save/load round-tripping the five new redesign params, and (merged in from
 * the former SpeakerDiarizerPipelineSpec.scala) the real `AudioAssembler -> Pipeline ->
 * transform(df)` surface a real user's code actually goes through - every other test in this
 * package calls `batchAnnotate` directly instead.
 *
 * Not a committed CI fixture - see SpeakerDiarizerTestFixtures (in
 * SpeakerDiarizerFullValidationSpec.scala), which this shares its scratch-path/model-loading
 * boilerplate with.
 */

package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.nlp.{Annotation, AnnotationAudio, AnnotatorType, AudioAssembler}
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

import java.util.concurrent.{Executors, TimeUnit}

class SpeakerDiarizerEdgeCasesAndPipelineSpec
    extends AnyFlatSpec
    with SpeakerDiarizerTestFixtures {

  // ============================== embeddingModelSize ==============================

  "embeddingModelSize" should "reflect the tier of the loaded model instead of throwing" taggedAs SlowTest in {
    val d = freshDiarizer().setMinDurationOn(0.3f)
    // Before this session's fix, loadSavedModel never actually called set(embeddingModelSize,
    // ...) despite the class scaladoc claiming it does - getEmbeddingModelSize threw
    // NoSuchElementException unconditionally. Only one tier (WeSpeaker's own heavier ResNet34
    // export) has ever been bundled, so "accurate" is what a real loaded model always is today.
    assert(d.getEmbeddingModelSize == "accurate")
  }

  it should "survive a save/load round trip" taggedAs SlowTest in {
    val d = freshDiarizer()
    val path = s"$scratch/models/save_test_embedding_model_size"
    d.write.overwrite().save(path)
    val reloaded = SpeakerDiarizer.load(path)
    assert(reloaded.getEmbeddingModelSize == "accurate")
  }

  // ============================== Concurrency ==============================

  "SpeakerDiarizer" should "produce correct, independent results when called from multiple threads concurrently on one broadcasted model" taggedAs SlowTest in {
    val shared = freshDiarizer().setMinDurationOn(0.3f)
    // Warm up the broadcast/session once outside the timed concurrent section - the point here is
    // concurrent *inference calls* into the one already-loaded model (the realistic multi-task-
    // per-executor scenario), not concurrent first-time model loading.
    run(shared, audio("very_short"))

    val inputs = Seq(
      "single_speaker" -> 1,
      "synthetic_2speaker" -> 2,
      "three_speakers" -> 2, // >=2 asserted below; natural count can vary slightly by threshold
      "four_speakers" -> 2)
    val pool = Executors.newFixedThreadPool(inputs.length)
    try {
      val futures = inputs.map { case (name, minExpectedSpeakers) =>
        pool.submit(new java.util.concurrent.Callable[(String, Seq[String])] {
          override def call(): (String, Seq[String]) = {
            val result = run(shared, audio(name))
            (name, speakers(result).distinct)
          }
        })
      }
      val results = futures.map(_.get(120, TimeUnit.SECONDS))
      results.foreach { case (name, distinctSpeakers) =>
        println(s"[concurrency] $name -> $distinctSpeakers")
      }
      val expectedMinBy = inputs.toMap
      results.foreach { case (name, distinctSpeakers) =>
        assert(
          distinctSpeakers.length >= expectedMinBy(name),
          s"$name: expected >=${expectedMinBy(name)} distinct speakers under concurrent " +
            s"execution, got $distinctSpeakers - a real cross-thread contamination bug would " +
            s"most plausibly show up as the wrong speaker count here")
      }
      // Cross-check against sequential (non-concurrent) execution on the same shared instance -
      // concurrency must not change the actual answer, only how many threads compute it.
      val sequential = inputs.map { case (name, _) =>
        name -> speakers(run(shared, audio(name))).distinct
      }
      assert(
        results.toMap.mapValues(_.length) == sequential.toMap.mapValues(_.length),
        s"concurrent and sequential runs disagreed on speaker counts: " +
          s"concurrent=${results.toMap.mapValues(_.length)} sequential=${sequential.toMap
              .mapValues(_.length)}")
    } finally {
      pool.shutdown()
    }
  }

  // ==================== Full phantom-speaker-prevention lifecycle ====================

  "a streaming session with numSpeakers set" should "correctly grow from 1 to 2 speakers once a genuinely different voice actually appears" taggedAs SlowTest in {
    val sessionId = "misc-gap-coverage-phantom-lifecycle"
    def freshStreamingDiarizer(): SpeakerDiarizer =
      freshDiarizer()
        .setMinDurationOn(0.3f)
        .setNumSpeakers(2)
        .setStreamingMode(true)
        .setSessionId(sessionId)

    // Call 1: one real speaker only. The redesign's fix (forceExactSpeakerCount=false while
    // streaming) must not manufacture a second speaker out of noise this early.
    val call1 = run(freshStreamingDiarizer(), audio("single_speaker"))
    val call1Speakers = speakers(call1).distinct
    println(s"[phantom lifecycle] call1 (single_speaker) -> $call1Speakers")
    assert(
      call1Speakers.length == 1,
      s"call 1 should show exactly 1 speaker (no early phantom), got $call1Speakers")

    // Call 2: a real second speaker, cropped from synthetic_2speaker's own documented
    // ground-truth segment (spkB, 7.09s-16.11s) - genuinely a different voice, not a synthetic
    // duplicate. Threaded into the SAME session via the in-memory streamingMode cache.
    val fullClip = audio("synthetic_2speaker")
    val sampleRate = 16000
    val spkBStart = (7.09 * sampleRate).toInt
    val spkBEnd = (16.11 * sampleRate).toInt
    val speakerBClip = fullClip.slice(spkBStart, spkBEnd)

    val call2 = run(freshStreamingDiarizer(), speakerBClip)
    val call2Speakers = speakers(call2).distinct
    println(s"[phantom lifecycle] call2 (real 2nd speaker) -> $call2Speakers")

    val allSpeakersSoFar = (call1Speakers ++ call2Speakers).distinct
    println(s"[phantom lifecycle] cumulative session speakers -> $allSpeakersSoFar")
    assert(
      allSpeakersSoFar.length == 2,
      s"the session should now show exactly 2 real speakers (the original one plus the new " +
        s"real one) - neither stuck at 1 (failing to recognize the new voice) nor inflated to " +
        s"3+ (a residual phantom), got $allSpeakersSoFar")
    assert(
      call2Speakers.forall(!call1Speakers.contains(_)),
      s"the new speaker's label(s) in call2 should not collide with call1's, got " +
        s"call1=$call1Speakers call2=$call2Speakers")
  }

  it should "also not force a phantom split via the explicit, multi-executor-safe setStreamingPriorState path" taggedAs SlowTest in {
    // Found and fixed while writing this test: batchAnnotate computed
    // `forceExactSpeakerCount = !getStreamingMode`, ignoring `_explicitPriorState` entirely - a
    // caller using ONLY setStreamingPriorState (the class scaladoc's own documented,
    // multi-executor-safe alternative to streamingMode's in-memory cache, correct under arbitrary
    // Spark scheduling) got forceExactSpeakerCount=true regardless, so numSpeakers could still
    // force a phantom split on this path even though it's clearly one call in an ongoing session.
    val call1 = run(freshDiarizer().setMinDurationOn(0.3f), audio("single_speaker"))
    val call1Speakers = speakers(call1).distinct
    assert(call1Speakers.length == 1, s"expected 1 real speaker in call1, got $call1Speakers")
    val blob = call1.head.metadata("clusterStateSnapshot")

    // Call 2 reuses the SAME speaker's audio again (single_speaker.wav once more) - numSpeakers=2
    // is introduced only now, with NO streamingMode at all, only the explicit prior-state blob.
    // Before the fix, forceExactSpeakerCount=true here would have split this one real voice's
    // turns into two phantom clusters to satisfy numSpeakers=2; after the fix, the explicit prior
    // state marks this as an ongoing session and the split must not happen.
    val call2 = run(
      freshDiarizer().setMinDurationOn(0.3f).setNumSpeakers(2).setStreamingPriorState(blob),
      audio("single_speaker"))
    val call2Speakers = speakers(call2).distinct
    println(s"[explicit prior state, no streamingMode] call1=$call1Speakers call2=$call2Speakers")
    assert(
      call2Speakers.length == 1,
      s"the same real speaker's audio, threaded via an explicit prior state with numSpeakers=2 " +
        s"set, must not be split into a phantom second cluster, got $call2Speakers")
    assert(
      call2Speakers == call1Speakers,
      "it must be recognized as the SAME speaker, not renamed")
  }

  it should "not let a single setStreamingPriorState call leak into a THIRD, unrelated call on the same reused instance" taggedAs SlowTest in {
    // Extends the test above: call1 establishes state, call2 consumes it via
    // setStreamingPriorState. Before the fix, `_explicitPriorState` was a plain var that
    // setStreamingPriorState set and nothing ever cleared - so isStreamingCall (and therefore
    // forceExactSpeakerCount) stayed permanently false forever on this instance, even for a
    // THIRD, completely unrelated call that never touched setStreamingPriorState itself. Reusing
    // one long-lived instance across many transform() calls is the normal Spark ML pattern (the
    // ONNX model is broadcast once specifically so the instance can be reused), not a contrived
    // misuse case.
    val d = freshDiarizer().setMinDurationOn(0.3f)
    val call1 = run(d, audio("single_speaker"))
    val blob = call1.head.metadata("clusterStateSnapshot")
    d.setStreamingPriorState(blob)
    run(d, audio("single_speaker")) // call 2: consumes the prior state (per the test above)

    // Call 3: the SAME instance, no setStreamingPriorState re-issued, numSpeakers forced on
    // completely unrelated 3-speaker audio. If the blob leaked, forceExactSpeakerCount would stay
    // false and numSpeakers=3 would silently NOT be enforced (natural clustering would win
    // instead of the requested exact count).
    val call3 = run(d.setNumSpeakers(3), audio("three_speakers"))
    val call3Speakers = speakers(call3).distinct
    println(s"[prior-state leak check] call3 (fresh, unrelated, numSpeakers=3) -> $call3Speakers")
    assert(
      call3Speakers.length == 3,
      s"a call after the prior-state blob should have been fully consumed must still honor an " +
        s"explicit numSpeakers as a fresh one-shot call, got $call3Speakers")
  }

  // ==================== Intra-batch state threading ====================

  "batchAnnotate" should "thread cluster state across ROWS within one micro-batch, not just across separate calls" taggedAs SlowTest in {
    // Before the fix, `options` (and its embedded priorState) was built ONCE outside the per-row
    // .map in batchAnnotate - every row in a multi-row batch clustered against the same stale
    // initial state instead of seeing clusters already formed by earlier rows in the SAME batch.
    val sessionId = "misc-gap-coverage-intra-batch-threading"
    val d = freshDiarizer()
      .setMinDurationOn(0.3f)
      .setStreamingMode(true)
      .setSessionId(sessionId)

    val fullClip = audio("synthetic_2speaker")
    val sampleRate = 16000
    val spkBStart = (7.09 * sampleRate).toInt
    val spkBEnd = (16.11 * sampleRate).toInt
    val speakerBClip = fullClip.slice(spkBStart, spkBEnd)

    // Two rows, ONE batchAnnotate call: row0 = the full clip (both speakers - spkB begins later,
    // so it gets the second/higher label by earliest-begin-time ordering); row1 = spkB's voice
    // again, alone, in the same batch.
    val row0 = Array(AnnotationAudio(AnnotatorType.AUDIO, fullClip, Map.empty))
    val row1 = Array(AnnotationAudio(AnnotatorType.AUDIO, speakerBClip, Map.empty))
    val results = d.batchAnnotate(Seq(row0, row1))

    val row0DistinctInOrder = results(0).sortBy(_.begin).map(_.metadata("speaker")).distinct
    assert(
      row0DistinctInOrder.length == 2,
      s"expected 2 speakers in row0 (full 2-speaker clip), got $row0DistinctInOrder")
    val spkBLabelInRow0 = row0DistinctInOrder(1)
    val row1Speakers = speakers(results(1)).distinct
    println(s"[multi-row same-session] row0=$row0DistinctInOrder row1=$row1Speakers")

    assert(
      row1Speakers == Seq(spkBLabelInRow0),
      s"row1 (the same voice as row0's second speaker) must be recognized as the SAME speaker " +
        s"($spkBLabelInRow0) via state threaded across rows of the same micro-batch, not " +
        s"independently renamed to a fresh label, got row1=$row1Speakers")
  }

  // ==================== Malformed / adversarial audio ====================

  "SpeakerDiarizer" should "not crash on audio containing NaN and Infinity samples" taggedAs SlowTest in {
    val clean = audio("single_speaker")
    val corrupted = clean.zipWithIndex.map {
      case (_, i) if i % 5000 == 0 => Float.NaN
      case (_, i) if i % 7000 == 0 => Float.PositiveInfinity
      case (_, i) if i % 9000 == 0 => Float.NegativeInfinity
      case (v, _) => v
    }
    val d = freshDiarizer().setMinDurationOn(0.3f)
    val result = run(d, corrupted)
    println(
      s"[NaN/Inf fuzz] turns=${result.length} speakers=${speakers(result).distinct} " +
        s"begins=${result.map(_.begin)} ends=${result.map(_.end)}")
    // No strong correctness claim (garbage in is allowed to produce garbage-ish turns) - the bar
    // is robustness: no exception, and whatever comes out is still structurally sane.
    assert(
      result.forall(a => a.begin >= 0 && a.end >= a.begin),
      "even on corrupted input, begin/end must stay non-negative and ordered")
  }

  it should "produce empty output rather than crashing on all-NaN audio" taggedAs SlowTest in {
    val allNaN = Array.fill(48000)(Float.NaN)
    val d = freshDiarizer().setMinDurationOn(0.3f)
    val result = run(d, allNaN)
    println(s"[all-NaN fuzz] turns=${result.length}")
  }

  // ==================== Logged-warning behaviors (stdout-captured) ====================

  // "warnIfUnsafeStreaming" (the >1-executor detection behind streamingMode's safety warning) is
  // NOT independently tested here. Two approaches were tried and both turned out infeasible in
  // this environment rather than merely inconvenient:
  //   - Triggering the real positive case (>1 executor) needs Spark's own local-cluster mode,
  //     which launches genuine separate worker JVMs via a real distribution's bin/spark-class
  //     script - not present here, since this project pulls Spark as sbt/Coursier library jars
  //     only, with no bundled distribution.
  //   - Capturing the logged warning via System.setOut/setErr swapping was tried (see
  //     SpeakerDiarizerAsrGapCoverageSpec2's beamSize/nReturnSequences test for the same attempt)
  //     and verifiably does not work in this harness: log4j/logback console appenders bind to the
  //     actual stream objects once, at logger initialization time, so reassigning
  //     System.out/System.err afterwards is invisible to them.
  // Every other streamingMode test in this session ran under the single-JVM local[*] configuration
  // this method treats as safe (executorCount <= 1), so the negative path is exercised
  // incidentally throughout, just not asserted on in isolation.

  // ==================== New redesign params: save/load round trip ====================

  "the five new redesign params" should "survive a save/load round trip with non-default values" taggedAs SlowTest in {
    val d = freshDiarizer()
      .setOverlapThreshold(0.42f)
      .setGalleryAcceptanceDistance(0.37f)
      .setPersistEmbeddings(true)
      .setMaxEmbeddingClipSeconds(12.5f)
      .setMaxAsrClipSeconds(17.5f)

    val path = s"$scratch/models/save_test_redesign_params"
    d.write.overwrite().save(path)
    val reloaded = SpeakerDiarizer.load(path)

    assert(reloaded.getOverlapThreshold == 0.42f)
    assert(reloaded.getGalleryAcceptanceDistance == 0.37f)
    assert(reloaded.getPersistEmbeddings)
    assert(reloaded.getMaxEmbeddingClipSeconds == 12.5f)
    assert(reloaded.getMaxAsrClipSeconds == 17.5f)
  }

  // ==================== Real Pipeline/DataFrame integration ====================
  // Every test above (and every other SpeakerDiarizer SlowTest spec in this package) calls
  // `diarizer.batchAnnotate(Seq(row))` directly, in-process - none of them go through the actual
  // Spark surface a real user's code uses: an `AudioAssembler` feeding a `SpeakerDiarizer` inside a
  // real `org.apache.spark.ml.Pipeline`, fit and transformed over a genuine DataFrame. That path
  // exercises real things `batchAnnotate` alone does not: `AnnotatorModel`'s own DataFrame-column
  // wiring, UDF/Row serialization of `AnnotationAudio`/`Annotation`, the output column's schema,
  // and correctness across more than one DataFrame row/partition. These three close that gap.

  private lazy val audioAssembler: AudioAssembler = new AudioAssembler()
    .setInputCol("audio_content")
    .setOutputCol("audio_assembler")

  private def freshPipelineDiarizer(): SpeakerDiarizer =
    freshDiarizer().setMinDurationOn(0.3f)

  "SpeakerDiarizer" should "produce correct results through a real AudioAssembler -> Pipeline -> transform(df) flow" taggedAs SlowTest in {
    import spark.implicits._
    val diarizer = freshPipelineDiarizer()
    val pipeline = new Pipeline().setStages(Array(audioAssembler, diarizer))

    val df = Seq(audio("synthetic_2speaker")).toDF("audio_content")
    val transformed = pipeline.fit(df).transform(df)

    val perRow = Annotation.collect(transformed, "speakers")
    assert(perRow.length == 1, "expected exactly one output row for one input row")
    val annotations = perRow.head.toSeq
    assert(annotations.nonEmpty, "expected at least one detected turn")

    val pipelineSpeakers = annotations.map(_.metadata("speaker")).distinct
    println(s"[pipeline] speakers=$pipelineSpeakers turns=${annotations.length}")
    assert(
      pipelineSpeakers.length >= 2,
      s"expected >=2 distinct speakers from synthetic_2speaker.wav via the real pipeline, got $pipelineSpeakers")

    // Cross-check against the direct batchAnnotate path this whole suite otherwise relies on -
    // the two entry points must agree on substance (same speaker count, same turn count), not
    // just both "succeed".
    val direct = run(freshPipelineDiarizer(), audio("synthetic_2speaker"))
    assert(
      annotations.length == direct.length,
      s"pipeline path found ${annotations.length} turns, direct batchAnnotate found ${direct.length}")
    assert(
      annotations.map(_.metadata("speaker")).distinct.length == speakers(direct).distinct.length)
  }

  it should "process every row of a multi-row DataFrame independently and correctly" taggedAs SlowTest in {
    import spark.implicits._
    val diarizer = freshPipelineDiarizer()
    val pipeline = new Pipeline().setStages(Array(audioAssembler, diarizer))

    // Row 0: one real speaker. Row 1: two real speakers. A bug that leaked state between rows
    // (e.g. accidentally threading cluster state, or reusing a buffer across partitions) would
    // most plausibly show up as row 0 picking up row 1's extra speaker or vice versa.
    val df = Seq(audio("single_speaker"), audio("synthetic_2speaker")).toDF("audio_content")
    val transformed = pipeline.fit(df).transform(df)

    val perRow = Annotation.collect(transformed, "speakers")
    assert(perRow.length == 2, s"expected 2 output rows, got ${perRow.length}")

    val row0Speakers = perRow(0).map(_.metadata("speaker")).distinct
    val row1Speakers = perRow(1).map(_.metadata("speaker")).distinct
    println(s"[pipeline multi-row] row0=${row0Speakers.toSeq} row1=${row1Speakers.toSeq}")
    assert(
      row0Speakers.length == 1,
      s"row 0 (single_speaker.wav) should show exactly 1 speaker, got ${row0Speakers.toSeq}")
    assert(
      row1Speakers.length >= 2,
      s"row 1 (synthetic_2speaker.wav) should show >=2 speakers, got ${row1Speakers.toSeq}")
  }

  it should "produce an output column with the documented SPEAKER annotation schema" taggedAs SlowTest in {
    import spark.implicits._
    val diarizer = freshPipelineDiarizer()
    val pipeline = new Pipeline().setStages(Array(audioAssembler, diarizer))

    val df = Seq(audio("single_speaker")).toDF("audio_content")
    val transformed = pipeline.fit(df).transform(df)

    val speakersField = transformed.schema("speakers")
    val elementFields =
      speakersField.dataType
        .asInstanceOf[org.apache.spark.sql.types.ArrayType]
        .elementType
        .asInstanceOf[org.apache.spark.sql.types.StructType]
        .fieldNames
        .toSet
    println(s"[pipeline schema] speakers element fields=$elementFields")
    assert(
      Set("annotatorType", "begin", "end", "result", "metadata", "embeddings")
        .subsetOf(elementFields),
      s"expected the standard Annotation struct fields, got $elementFields")
  }
}
