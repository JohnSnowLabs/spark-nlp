/*
 * The repetitionPenalty/noRepeatNgramSize dead-parameter bug found and fixed this session lives
 * in Whisper.scala, which both SpeakerDiarizer and WhisperForCTC share - the fix was verified
 * end-to-end through SpeakerDiarizer's path (SpeakerDiarizerAsrGapCoverageSpec2), but never
 * through WhisperForCTC directly. WhisperForCTCTest itself only covers this via `.pretrained()`
 * (a network download), which this file avoids by reusing the same local ONNX Whisper export
 * already used elsewhere in this session (loadSavedModel, no network).
 *
 * Deliberately kept as its own small file rather than folded into the SpeakerDiarizer suites it
 * mirrors (which share a SpeakerDiarizerTestFixtures trait) or into the pre-existing
 * WhisperForCTCTest.scala: it tests a genuinely different class (WhisperForCTC directly, not via
 * SpeakerDiarizer) through a genuinely different loading path (local ONNX export, not
 * `.pretrained()`) than either of those - the fixture shapes don't actually overlap enough to
 * share, and every other SpeakerDiarizer-adjacent SlowTest file in this package has already been
 * shown (see SpeakerDiarizerAsrGapCoverageSpec2's header) to risk a real OOM once too many
 * separate Whisper-bundled-model files start sharing one JVM.
 *
 * Not a committed CI fixture - same SPARKNLP_SPEAKER_DIARIZER_TEST_SCRATCH override point as
 * SpeakerDiarizerTestFixtures (in SpeakerDiarizerFullValidationSpec.scala), so one environment
 * variable repoints every SlowTest file in this package at once.
 */

package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotationAudio, AnnotatorType}
import com.johnsnowlabs.tags.SlowTest
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source

class WhisperForCTCLocalRepetitionParamsSpec extends AnyFlatSpec {

  private val scratch = sys.env.getOrElse(
    "SPARKNLP_SPEAKER_DIARIZER_TEST_SCRATCH",
    "/private/tmp/claude-501/-Users-abdullah-Documents-spark-nlp--claude-worktrees-opt-125m-research-90713f/61bcb16c-f914-49ff-af4a-c56c4c5e7ac6/scratchpad")
  private val whisperPkgPath = s"$scratch/models/whisper_pkg"
  private val td = s"$scratch/models/testdata"
  private val spark = ResourceHelper.spark

  private def loadFloats(path: String): Array[Float] = {
    val src = Source.fromFile(path)
    try {
      src.getLines().map(_.trim.toFloat).toArray
    } finally src.close()
  }

  private def audio(name: String): Array[Float] = loadFloats(s"$td/${name}_floats.txt")

  private def freshModel(): WhisperForCTC =
    WhisperForCTC
      .loadSavedModel(whisperPkgPath, spark)
      .setInputCols("audio_assembler")
      .setOutputCol("document")

  private def run(model: WhisperForCTC, samples: Array[Float]): Seq[Annotation] = {
    val row = Array(AnnotationAudio(AnnotatorType.AUDIO, samples, Map.empty))
    model.batchAnnotate(Seq(row)).head
  }

  private def text(anns: Seq[Annotation]): String = anns.map(_.result).mkString(" ")

  // Same manufactured verbatim-repeat trick used for SpeakerDiarizer: single_speaker.wav
  // concatenated with itself, well under Whisper's 30s window, guaranteed to make the model
  // transcribe the same sentence twice - real material for repetitionPenalty to act on.
  private lazy val repeatedSpeech: Array[Float] = {
    val clip = audio("single_speaker")
    clip ++ clip
  }

  "WhisperForCTC.setRepetitionPenalty" should "change greedy output on speech that verbatim-repeats itself" taggedAs SlowTest in {
    val baseline = run(freshModel().setRepetitionPenalty(1.0), repeatedSpeech)
    val penalized = run(freshModel().setRepetitionPenalty(1.8), repeatedSpeech)

    println(s"[WhisperForCTC repetitionPenalty] baseline=${text(baseline)}")
    println(s"[WhisperForCTC repetitionPenalty] penalized=${text(penalized)}")
    assert(text(baseline).nonEmpty && text(penalized).nonEmpty)
    assert(
      text(baseline) != text(penalized),
      "the same fix verified through SpeakerDiarizer must also take effect through " +
        "WhisperForCTC's own batchAnnotate path, since both share Whisper.getLogitProcessors")
  }

  it should "leave output unchanged at the default penalty of 1.0" taggedAs SlowTest in {
    val a = run(freshModel(), repeatedSpeech)
    val b = run(freshModel().setRepetitionPenalty(1.0), repeatedSpeech)
    assert(text(a) == text(b))
  }

  "repetitionPenalty/noRepeatNgramSize" should "survive a WhisperForCTC save/load round trip" taggedAs SlowTest in {
    // Both are plain Spark Params (no custom Feature/IO code), so this should just work via the
    // standard params.json mechanism - unverified until now, since every other test in this file
    // (and the SpeakerDiarizer-side fix verification) only ever set these on a freshly-loaded
    // model, never round-tripped one through .save()/.load().
    val original = freshModel().setRepetitionPenalty(1.8).setNoRepeatNgramSize(3)
    val path = s"$scratch/models/save_test_whisperctc_repetition_params"
    original.write.overwrite().save(path)
    val reloaded = WhisperForCTC.load(path)

    assert(reloaded.getRepetitionPenalty == 1.8)
    assert(reloaded.getNoRepeatNgramSize == 3)

    // Also confirm the reloaded model's own inference still reflects the restored penalty value,
    // not just that the Param getter reports the right number.
    val reloadedResult =
      run(reloaded.setInputCols("audio_assembler").setOutputCol("document"), repeatedSpeech)
    val baseline = run(freshModel().setRepetitionPenalty(1.0), repeatedSpeech)
    println(s"[WhisperForCTC save/load] reloaded=${text(reloadedResult)}")
    assert(
      text(reloadedResult) != text(baseline),
      "the reloaded model must actually apply the restored repetitionPenalty during inference, " +
        "not just report the right value from a getter")
  }
}
