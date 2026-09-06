/*
 * ASR generation-parameter validation for SpeakerDiarizer, split into its own file/JVM: loading
 * several full Whisper-bundled diarizer instances back to back in the same JVM as
 * SpeakerDiarizerFullValidationSpec's other tests reliably exhausts the 4GB test heap even
 * though each test individually is cheap - broadcasts from prior tests are never unpersisted
 * within one Spark session. Splitting this out is what makes the suite runnable at all, not an
 * optional cleanup.
 *
 * A merge of this file with SpeakerDiarizerAsrGapCoverageSpec + SpeakerDiarizerAsrGapCoverageSpec2
 * (13 combined tests) was tried and measured directly: it passed once as an isolated probe, then
 * OOM-killed (exit 137) on both of the next two real attempts, dying at the same test
 * ("randomSeed", the 12th of 13) each time - genuine, reproducible flakiness at this heap size,
 * not a one-off fluke. Reverted back to three separate files rather than ship an intermittently-
 * crashing test suite for the sake of a smaller file count. See SpeakerDiarizerAsrGapCoverageSpec
 * and SpeakerDiarizerAsrGapCoverageSpec2 for the rest.
 *
 * Not a committed CI fixture - see SpeakerDiarizerTestFixtures (in
 * SpeakerDiarizerFullValidationSpec.scala), which this shares its scratch-path/model-loading
 * boilerplate with.
 */

package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.tags.SlowTest
import org.scalatest.flatspec.AnyFlatSpec

class SpeakerDiarizerAsrGenerationParamsSpec
    extends AnyFlatSpec
    with SpeakerDiarizerTestFixtures {

  "setMaxOutputLength" should "actually truncate transcripts when set very small" taggedAs SlowTest in {
    val short = freshDiarizerWithAsr().setMinDurationOn(0.3f).setMaxOutputLength(8)
    val result = run(short, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    val shortLen = result.map(_.result.length).sum
    println(s"[maxOutputLength=8] totalChars=$shortLen texts=${result.map(_.result)}")
    // Compared against the known unrestricted transcript from earlier full-suite runs (each
    // segment routinely 60-100+ chars) - 8 output tokens should produce a visibly short result.
    assert(shortLen < 200, s"expected a heavily truncated transcript, got $shortLen chars total")
  }

  "setDoSample/setTemperature" should "be reachable without erroring (sampled generation)" taggedAs SlowTest in {
    val sampled = freshDiarizerWithAsr()
      .setMinDurationOn(0.3f)
      .setDoSample(true)
      .setTemperature(0.8)
      .setRandomSeed(42L)
    val result = run(sampled, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    result.foreach(a => println(s"[doSample=true] ${a.metadata("speaker")}: ${a.result}"))
  }

  "greedy decoding (doSample=false, the default)" should "be deterministic across repeated ASR runs" taggedAs SlowTest in {
    val d1 = freshDiarizerWithAsr().setMinDurationOn(0.3f)
    val r1 = run(d1, audio("synthetic_2speaker"))
    val d2 = freshDiarizerWithAsr().setMinDurationOn(0.3f)
    val r2 = run(d2, audio("synthetic_2speaker"))
    assert(r1.map(_.result) == r2.map(_.result), s"ASR text should be deterministic: $r1 vs $r2")
  }

  "setAsrTask/setAsrLanguage" should "be settable and reach the ASR call without erroring" taggedAs SlowTest in {
    val d = freshDiarizerWithAsr()
      .setMinDurationOn(0.3f)
      .setAsrTask("<|transcribe|>")
      .setAsrLanguage("<|en|>")
    val result = run(d, audio("synthetic_2speaker"))
    assert(result.nonEmpty)
    assert(
      result.forall(_.result.trim.nonEmpty),
      "expected transcripts with an explicit language/task set")
  }

  it should "reject a malformed asrTask value" taggedAs SlowTest in {
    assertThrows[IllegalArgumentException] {
      freshDiarizer().setAsrTask("bogus")
    }
  }

  it should "reject a malformed asrLanguage value" taggedAs SlowTest in {
    assertThrows[IllegalArgumentException] {
      freshDiarizer().setAsrLanguage("english")
    }
  }
}
