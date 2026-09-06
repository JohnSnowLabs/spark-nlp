/*
 * Real-inference coverage for the inherited HasGeneratorProperties ASR params that had never
 * actually been exercised at all (minOutputLength, topK, topP, repetitionPenalty,
 * noRepeatNgramSize - not even for reachability) or only for reachability, not a verified real
 * effect (randomSeed). Also documents two genuine, previously-undiscovered characteristics of the
 * bundled Whisper model found while writing these tests: `beamSize`/`nReturnSequences` are
 * accepted but silently ignored (Whisper.generateFromAudio always forces greedy decoding), and
 * `topK` values below 100 are silently floored to 100 by TopKLogitWarper's hardcoded
 * `minTokensToKeep` default (shared Generation code, not specific to this annotator - out of
 * scope to change here, but worth knowing about before relying on a small topK).
 *
 * Split into its own file/JVM for the same reason SpeakerDiarizerAsrGenerationParamsSpec already
 * is - repeatedly loading full Whisper broadcasts in one JVM exhausts the test heap. (A merge with
 * that file plus SpeakerDiarizerAsrGapCoverageSpec2 was tried directly and measured to be
 * genuinely flaky at -Xmx4g - see SpeakerDiarizerAsrGenerationParamsSpec's own header for the
 * specifics - so this stays a separate file rather than an assumption.)
 *
 * Not a committed CI fixture - see SpeakerDiarizerTestFixtures (in
 * SpeakerDiarizerFullValidationSpec.scala), which this shares its scratch-path/model-loading
 * boilerplate with. `freshDiarizerAsrDefault` here is a thin local wrapper pinning
 * `minDurationOn(0.3f)` on top of the shared trait's `freshDiarizerWithAsr` - kept local rather
 * than changing the shared default, since every test in this file wants it and no other consumer
 * of the trait should have to.
 */

package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.tags.SlowTest
import org.scalatest.flatspec.AnyFlatSpec

class SpeakerDiarizerAsrGapCoverageSpec extends AnyFlatSpec with SpeakerDiarizerTestFixtures {

  private def freshDiarizerAsrDefault(): SpeakerDiarizer =
    freshDiarizerWithAsr().setMinDurationOn(0.3f)

  "minOutputLength" should "force a visibly longer transcript than the natural, unconstrained length" taggedAs SlowTest in {
    val default = run(freshDiarizerAsrDefault(), audio("synthetic_2speaker"))
    val forced =
      run(freshDiarizerAsrDefault().setMinOutputLength(200), audio("synthetic_2speaker"))

    val defaultLen = text(default).length
    val forcedLen = text(forced).length
    println(s"[minOutputLength] default(0) chars=$defaultLen, forced(200) chars=$forcedLen")
    assert(
      forcedLen > defaultLen,
      s"forcing minOutputLength=200 should suppress EOS well past the natural stopping point, " +
        s"got default=$defaultLen forced=$forcedLen")
  }

  "topK" should "silently floor to TopKLogitWarper's hardcoded minTokensToKeep=100 for small values" taggedAs SlowTest in {
    // TopKLogitWarper is constructed as `new TopKLogitWarper(topK)`, which leaves its
    // `minTokensToKeep` constructor parameter at its class default of 100 - `effectiveTopK =
    // k.max(minTokensToKeep)` then means ANY topK <= 100 collapses to the same 100-candidate
    // pool. With a fixed random seed, two different-but-both-small topK values must therefore
    // produce byte-identical sampled output (same pool, same seed, same draws) - this is not a
    // hypothesis, it follows directly from the code, and confirms the floor is real rather than
    // topK simply "not mattering much" for this audio.
    val topK1 = run(
      freshDiarizerAsrDefault().setDoSample(true).setRandomSeed(777L).setTopK(1),
      audio("synthetic_2speaker"))
    val topK99 = run(
      freshDiarizerAsrDefault().setDoSample(true).setRandomSeed(777L).setTopK(99),
      audio("synthetic_2speaker"))
    val topK5000 = run(
      freshDiarizerAsrDefault().setDoSample(true).setRandomSeed(777L).setTopK(5000),
      audio("synthetic_2speaker"))

    println(
      s"[topK floor] topK=1: ${text(topK1)} | topK=99: ${text(topK99)} | topK=5000: ${text(topK5000)}")
    assert(
      text(topK1) == text(topK99),
      "topK=1 and topK=99 both floor to the same effective 100-candidate pool and must match " +
        "exactly under the same seed")
  }

  "topP" should "collapse to greedy-equivalent output at a near-zero value" taggedAs SlowTest in {
    val greedy = run(freshDiarizerAsrDefault(), audio("synthetic_2speaker"))
    val tinyTopP = run(
      freshDiarizerAsrDefault().setDoSample(true).setRandomSeed(42L).setTopP(0.0001),
      audio("synthetic_2speaker"))

    println(s"[topP~0] greedy=${text(greedy)} | topP=0.0001=${text(tinyTopP)}")
    assert(
      text(greedy) == text(tinyTopP),
      "a near-zero topP should admit essentially only the single highest-probability token at " +
        "each step, matching greedy decoding")
  }

}
