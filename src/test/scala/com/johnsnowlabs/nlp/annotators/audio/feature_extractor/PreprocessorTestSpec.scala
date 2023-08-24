package com.johnsnowlabs.nlp.annotators.audio.feature_extractor

import com.johnsnowlabs.util.TestUtils.{readFile, tolerantFloatEq}
import org.scalatest.flatspec.AnyFlatSpec
class PreprocessorTestSpec extends AnyFlatSpec {

  val rawFloats: Array[Float] =
    readFile("src/test/resources/audio/txt/librispeech_asr_0.txt").split("\n").map(_.toFloat)

  val preprocessor: WhisperPreprocessor = {
    val ppPath = "src/test/resources/audio/configs/preprocessor_config_whisper.json"
    val ppJsonString = readFile(ppPath)

    Preprocessor.loadPreprocessorConfig(ppJsonString).asInstanceOf[WhisperPreprocessor]
  }

  behavior of "AudioPreprocessor"

  it should "normalize" in {
    val normalized = Preprocessor.normalize(rawFloats)
    assert(Preprocessor.mean(normalized) === 0f)
    assert(Preprocessor.variance(normalized).toFloat === 1f)
  }

  it should "pad" in {
    val padded = Preprocessor.pad(
      rawFloats,
      preprocessor.padding_value,
      preprocessor.n_samples,
      preprocessor.padding_side)

    assert(padded.length == preprocessor.n_samples)
  }

}
