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

package com.johnsnowlabs.nlp.annotators.audio.feature_extractor

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source

/** Validates [[KaldiFbank]] against a reference computed with
  * `torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10,
  * dither=0.0, sample_frequency=16000)` on 1 second of fixed-seed noise — see
  * `fbank_test_signal.csv` / `fbank_reference_torchaudio.csv` for the fixtures.
  */
class KaldiFbankTestSpec extends AnyFlatSpec {

  private def loadFloats(path: String): Array[Float] =
    Source.fromFile(path).getLines().map(_.trim.toFloat).toArray

  private def loadMatrix(path: String): Array[Array[Float]] =
    Source
      .fromFile(path)
      .getLines()
      .map(_.split(",").map(_.trim.toFloat))
      .toArray

  "KaldiFbank" should "match torchaudio.compliance.kaldi.fbank within tolerance" taggedAs FastTest in {
    val signal = loadFloats("src/test/resources/audio/csv/fbank_test_signal.csv")
    val reference = loadMatrix("src/test/resources/audio/csv/fbank_reference_torchaudio.csv")

    val fbank = new KaldiFbank(sampleFrequency = 16000, numMelBins = 80)
    val actual = fbank.extractFeatures(signal)

    assert(
      actual.length == reference.length,
      s"frame count mismatch: got ${actual.length}, expected ${reference.length}")
    assert(actual.head.length == reference.head.length)

    var maxAbsDiff = 0.0
    var sumAbsDiff = 0.0
    var count = 0
    for (i <- actual.indices; j <- actual(i).indices) {
      val diff = math.abs(actual(i)(j) - reference(i)(j))
      maxAbsDiff = math.max(maxAbsDiff, diff)
      sumAbsDiff += diff
      count += 1
    }
    val meanAbsDiff = sumAbsDiff / count

    // log-mel energies for this signal range roughly [-9, 2]; a mean/max tolerance well under
    // that range confirms the implementations agree, not just happen to be the same order of
    // magnitude.
    assert(meanAbsDiff < 0.05, s"mean abs diff too high: $meanAbsDiff")
    assert(maxAbsDiff < 0.5, s"max abs diff too high: $maxAbsDiff")
  }

  it should "drop a trailing partial frame rather than pad it (snip_edges=true)" taggedAs FastTest in {
    val fbank = new KaldiFbank(sampleFrequency = 16000, numMelBins = 80)
    // 400 samples = exactly one 25ms frame, plus a partial remainder shorter than one frame shift
    val samples = Array.fill(400 + 50)(0.01f)
    val features = fbank.extractFeatures(samples)
    assert(features.length == 1)
  }

  it should "return empty output for audio shorter than one frame" taggedAs FastTest in {
    val fbank = new KaldiFbank(sampleFrequency = 16000, numMelBins = 80)
    assert(fbank.extractFeatures(Array.fill(100)(0.01f)).isEmpty)
  }

  "KaldiFbank.extractNormalizedFeatures" should "zero-mean each dimension across time" taggedAs FastTest in {
    val signal = loadFloats("src/test/resources/audio/csv/fbank_test_signal.csv")
    val normalized = KaldiFbank.extractNormalizedFeatures(signal)
    val dim = normalized.head.length
    for (d <- 0 until dim) {
      val mean = normalized.map(_(d)).sum / normalized.length
      assert(math.abs(mean) < 1e-3, s"dimension $d mean not ~0: $mean")
    }
  }

  it should "cache instances per (sampleFrequency, numMelBins) without cross-contaminating between different configs" taggedAs FastTest in {
    // extractNormalizedFeatures now reuses a KaldiFbank instance cached by (sampleFrequency,
    // numMelBins) instead of constructing one fresh on every call (rebuilding the mel filterbank
    // and Povey window is real, sample-independent setup work otherwise redone per turn). This
    // interleaves two different configs and returns to the first, verifying the cache key
    // actually distinguishes them - a wrong/collapsed key would let one config's cached instance
    // silently serve the other's request.
    val signal = loadFloats("src/test/resources/audio/csv/fbank_test_signal.csv")

    val configA =
      KaldiFbank.extractNormalizedFeatures(signal, sampleFrequency = 16000, numMelBins = 80)
    val configB =
      KaldiFbank.extractNormalizedFeatures(signal, sampleFrequency = 16000, numMelBins = 40)
    val configAAgain =
      KaldiFbank.extractNormalizedFeatures(signal, sampleFrequency = 16000, numMelBins = 80)

    assert(configA.head.length == 80, s"expected 80 mel bins, got ${configA.head.length}")
    assert(configB.head.length == 40, s"expected 40 mel bins, got ${configB.head.length}")
    assert(
      configA.length == configAAgain.length &&
        configA.zip(configAAgain).forall { case (frameA, frameB) =>
          frameA.zip(frameB).forall { case (x, y) => math.abs(x - y) < 1e-6 }
        },
      "requesting the same (sampleFrequency, numMelBins) config again after a different config " +
        "was used in between must return identical results - the per-config instance cache must " +
        "not cross-contaminate between different cache keys")
  }
}
