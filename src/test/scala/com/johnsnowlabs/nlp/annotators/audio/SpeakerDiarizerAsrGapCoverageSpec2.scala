/*
 * Continuation of SpeakerDiarizerAsrGapCoverageSpec, split into its own file/JVM: loading several
 * full Whisper-bundled diarizer instances back to back in one JVM (repetitionPenalty,
 * noRepeatNgramSize, randomSeed, and beamSize/nReturnSequences together load ~9 of them) reliably
 * exhausts the 4GB test heap even though each individual test is cheap - confirmed by an actual
 * OOM kill combining all of these with minOutputLength/topK/topP in one file, and reconfirmed
 * later by a second, independent attempt to merge this file with
 * SpeakerDiarizerAsrGapCoverageSpec and SpeakerDiarizerAsrGenerationParamsSpec into one (passed
 * once as an isolated probe, then OOM-killed on both subsequent real attempts - see
 * SpeakerDiarizerAsrGenerationParamsSpec's own header). See SpeakerDiarizerAsrGapCoverageSpec's
 * own header for what's covered and why.
 *
 * Not a committed CI fixture - see SpeakerDiarizerTestFixtures (in
 * SpeakerDiarizerFullValidationSpec.scala), which this shares its scratch-path/model-loading
 * boilerplate with.
 */

package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.tags.SlowTest
import org.scalatest.flatspec.AnyFlatSpec

class SpeakerDiarizerAsrGapCoverageSpec2 extends AnyFlatSpec with SpeakerDiarizerTestFixtures {

  private def freshDiarizerAsrDefault(): SpeakerDiarizer =
    freshDiarizerWithAsr().setMinDurationOn(0.3f)

  // single_speaker.wav's own natural transcript ("He was in a fevered state... He would have to
  // pay her the money...") turned out to have essentially no literal repeated words/bigrams for
  // either param to act on - repetitionPenalty=1.8 and noRepeatNgramSize=2 both produced byte-
  // identical output to the baseline against it (a real, honest finding in its own right: greedy
  // decoding on clean, easy speech is robust enough that these params can be complete no-ops in
  // practice unless the natural output actually contains a repeat). Concatenating the clip with
  // itself (well under Whisper's 30s window) manufactures a real, verbatim repeat - the same
  // sentence back to back - giving both params real material to act on instead of asserting
  // against a text that happens to have none.
  private lazy val repeatedSpeech: Array[Float] = {
    val clip = audio("single_speaker")
    clip ++ clip
  }

  "repetitionPenalty" should "change greedy output on speech that verbatim-repeats itself" taggedAs SlowTest in {
    val baseline =
      run(freshDiarizerAsrDefault().setRepetitionPenalty(1.0), repeatedSpeech)
    val penalized =
      run(freshDiarizerAsrDefault().setRepetitionPenalty(1.8), repeatedSpeech)

    println(s"[repetitionPenalty] baseline=${text(baseline)} | penalized=${text(penalized)}")
    assert(text(baseline).nonEmpty && text(penalized).nonEmpty)
    assert(
      text(baseline) != text(penalized),
      "a strong repetition penalty should alter at least one word choice once the same " +
        "sentence genuinely repeats verbatim within one turn")
  }

  "noRepeatNgramSize" should "reach real ASR generation without erroring" taggedAs SlowTest in {
    // Word-for-word text repetition (confirmed above for repetitionPenalty, on this exact
    // manufactured clip) doesn't guarantee TOKEN-level bigram repetition: Whisper's byte-level BPE
    // can tokenize the same word differently depending on what precedes it, so this real clip did
    // not end up exercising an actual banned-bigram decision either way. The mechanism itself -
    // that NoRepeatNgramsLogitProcessor correctly bans the token that previously followed a
    // repeated token-id bigram - is verified deterministically and unconditionally in
    // LogitProcessorTest instead, which controls the exact token sequence directly rather than
    // depending on a specific tokenizer's real output. This just confirms wiring a nonzero value
    // through the whole SpeakerDiarizer -> DiarizationOptions -> Whisper chain doesn't error.
    val baseline = run(freshDiarizerAsrDefault(), repeatedSpeech)
    val constrained =
      run(freshDiarizerAsrDefault().setNoRepeatNgramSize(2), repeatedSpeech)

    println(s"[noRepeatNgramSize] baseline=${text(baseline)} | constrained=${text(constrained)}")
    assert(text(baseline).nonEmpty)
    assert(text(constrained).nonEmpty)
  }

  "randomSeed" should "make sampled generation reproducible across separate calls with the same seed" taggedAs SlowTest in {
    val run1 = run(
      freshDiarizerAsrDefault().setDoSample(true).setTemperature(0.9).setRandomSeed(2024L),
      audio("synthetic_2speaker"))
    val run2 = run(
      freshDiarizerAsrDefault().setDoSample(true).setTemperature(0.9).setRandomSeed(2024L),
      audio("synthetic_2speaker"))
    val run3 = run(
      freshDiarizerAsrDefault().setDoSample(true).setTemperature(0.9).setRandomSeed(99L),
      audio("synthetic_2speaker"))

    println(s"[randomSeed] seed=2024 run1=${text(run1)}")
    println(s"[randomSeed] seed=2024 run2=${text(run2)}")
    println(s"[randomSeed] seed=99   run3=${text(run3)}")
    assert(
      text(run1) == text(run2),
      "the same randomSeed must reproduce byte-identical sampled output across separate calls")
    // Not asserted strictly (a confident model can legitimately land on the same tokens under a
    // different seed too) - logged so a real divergence is visible when it happens, without
    // making the test flaky on a rare seed collision in the sampled path.
  }

  "beamSize/nReturnSequences" should "be accepted but have no effect on the bundled Whisper model's output" taggedAs SlowTest in {
    // Confirmed by reading Whisper.generateFromAudio directly: it unconditionally forces greedy
    // decoding and only logs a warning when beamSize > 1, regardless of what's requested - the
    // real beam-search code path (Generate.generate's BeamSearchScorer) is never reached for this
    // model. This documents that actual, verified behavior instead of leaving it undiscovered.
    val default = run(freshDiarizerAsrDefault(), audio("synthetic_2speaker"))

    // A System.setOut/setErr-based capture of the logged warning was tried here first, but
    // verifiably doesn't work in this harness: the assertion on captured content failed with
    // *empty* captured output even though the behavioral no-op assertions below (which don't
    // depend on capturing anything) passed correctly on the same run - meaning the warning really
    // was logged, just not through the stream objects this test swapped. This matches how
    // log4j/logback console appenders normally work: they bind directly to the actual
    // System.out/System.err PrintStream objects once, at logger initialization time (early in the
    // JVM's life, long before this test runs), so reassigning System.out/System.err afterwards
    // doesn't redirect anything already bound to the originals. Confirming the warning's exact
    // text would need a logger-framework-specific in-process appender instead - out of scope here.
    // The behavior this warning describes (greedy decoding regardless of beamSize) is what
    // actually matters and is verified directly below.
    val withBeamAndReturnSeqs = run(
      freshDiarizerAsrDefault().setBeamSize(4).setNReturnSequences(3),
      audio("synthetic_2speaker"))

    println(s"[beamSize/nReturnSequences no-op] default=${text(default)}")
    println(
      s"[beamSize/nReturnSequences no-op] beamSize=4,nReturnSequences=3=${text(withBeamAndReturnSeqs)}")
    assert(text(default) == text(withBeamAndReturnSeqs))
    assert(
      withBeamAndReturnSeqs.length == default.length,
      "nReturnSequences=3 must still produce exactly one transcript per turn, not three, since " +
        "the beam-search path it would otherwise control is never used")
  }
}
