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

import breeze.linalg.DenseVector
import breeze.math.Complex
import breeze.signal.fourierTr

/** Kaldi-compatible mel filterbank ("fbank") feature extraction, matching
  * `torchaudio.compliance.kaldi.fbank`'s defaults — the exact recipe WeSpeaker (and most
  * Kaldi-heritage x-vector/speaker-embedding models) train and infer with. This is a distinct
  * convention from the HuggingFace/Slaney-style mel spectrogram already in [[AudioUtils]] (used
  * by Whisper): different mel-scale formula, no Slaney energy normalization, a Povey window
  * instead of Hann, and per-frame DC removal + preemphasis before windowing — reusing
  * `AudioUtils` would silently produce the wrong features for a Kaldi-trained embedding model.
  *
  * Numerically validated against `torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80,
  * frame_length=25, frame_shift=10, dither=0.0, sample_frequency=16000)` (all other params at
  * torchaudio's defaults: `window_type="povey"`, `preemphasis_coefficient=0.97`,
  * `remove_dc_offset=true`, `round_to_power_of_two=true`, `snip_edges=true`).
  *
  * @param sampleFrequency
  *   audio sampling rate in Hz
  * @param numMelBins
  *   number of mel filterbank output bins
  * @param frameLengthMs
  *   analysis window length, in milliseconds
  * @param frameShiftMs
  *   hop between analysis windows, in milliseconds
  * @param preemphasisCoefficient
  *   first-order preemphasis filter coefficient
  * @param lowFreq
  *   lowest mel-filter edge, in Hz
  * @param highFreq
  *   highest mel-filter edge, in Hz; `<= 0` means `nyquist + highFreq` (`0` = exactly Nyquist),
  *   matching Kaldi's convention
  */
class KaldiFbank(
    sampleFrequency: Int = 16000,
    numMelBins: Int = 80,
    frameLengthMs: Float = 25.0f,
    frameShiftMs: Float = 10.0f,
    preemphasisCoefficient: Double = 0.97,
    lowFreq: Double = 20.0,
    highFreq: Double = 0.0) {

  private val frameLength: Int = math.round(sampleFrequency * frameLengthMs / 1000.0f)
  private val frameShift: Int = math.round(sampleFrequency * frameShiftMs / 1000.0f)
  private val fftLength: Int = KaldiFbank.nextPowerOfTwo(frameLength)
  private val nyquist: Double = sampleFrequency / 2.0
  private val effectiveHighFreq: Double = if (highFreq <= 0.0) nyquist + highFreq else highFreq

  private val poveyWindow: Array[Double] = Array.tabulate(frameLength) { i =>
    val a = 2.0 * math.Pi / (frameLength - 1)
    math.pow(0.5 - 0.5 * math.cos(a * i), 0.85)
  }

  private case class MelFilter(startBin: Int, weights: Array[Double])
  private val melFilters: Array[MelFilter] = buildMelFilters()

  private def hzToMel(hz: Double): Double = 1127.0 * math.log(1.0 + hz / 700.0)

  private def buildMelFilters(): Array[MelFilter] = {
    val numFftBins = fftLength / 2 + 1
    val melLow = hzToMel(lowFreq)
    val melHigh = hzToMel(effectiveHighFreq)
    val melPoints = Array.tabulate(numMelBins + 2) { i =>
      melLow + (melHigh - melLow) * i / (numMelBins + 1)
    }
    val fftBinFreqs = Array.tabulate(numFftBins) { k => k * sampleFrequency.toDouble / fftLength }
    val fftBinMels = fftBinFreqs.map(hzToMel)

    Array.tabulate(numMelBins) { m =>
      val leftMel = melPoints(m)
      val centerMel = melPoints(m + 1)
      val rightMel = melPoints(m + 2)
      val weights = new Array[Double](numFftBins)
      var start = -1
      var end = -1
      var k = 0
      while (k < numFftBins) {
        val mel = fftBinMels(k)
        val w =
          if (mel > leftMel && mel < rightMel) {
            if (mel <= centerMel) (mel - leftMel) / (centerMel - leftMel)
            else (rightMel - mel) / (rightMel - centerMel)
          } else 0.0
        if (w > 0.0) {
          if (start == -1) start = k
          end = k
          weights(k) = w
        }
        k += 1
      }
      if (start == -1) MelFilter(0, Array.emptyDoubleArray)
      else MelFilter(start, weights.slice(start, end + 1))
    }
  }

  /** Extracts fbank features from `samples` (mono, at `sampleFrequency` Hz).
    *
    * @return
    *   `[numFrames][numMelBins]` log mel filterbank energies, or an empty array if `samples` is
    *   shorter than one frame (matches `snip_edges=true`: no zero-padding, partial frames at the
    *   end are dropped)
    */
  def extractFeatures(samples: Array[Float]): Array[Array[Float]] = {
    if (samples.length < frameLength) return Array.empty

    val numFrames = 1 + (samples.length - frameLength) / frameShift
    val epsilon = 1.1920929e-7

    Array.tabulate(numFrames) { frameIdx =>
      val offset = frameIdx * frameShift
      val frame = new Array[Double](frameLength)
      var i = 0
      while (i < frameLength) {
        frame(i) = samples(offset + i).toDouble
        i += 1
      }

      val mean = frame.sum / frameLength
      i = 0
      while (i < frameLength) {
        frame(i) -= mean
        i += 1
      }

      i = frameLength - 1
      while (i >= 1) {
        frame(i) -= preemphasisCoefficient * frame(i - 1)
        i -= 1
      }
      frame(0) -= preemphasisCoefficient * frame(0)

      i = 0
      while (i < frameLength) {
        frame(i) *= poveyWindow(i)
        i += 1
      }

      val padded = DenseVector.zeros[Complex](fftLength)
      i = 0
      while (i < frameLength) {
        padded(i) = Complex(frame(i), 0.0)
        i += 1
      }
      val spectrum = fourierTr(padded)
      val numFftBins = fftLength / 2 + 1
      val power = new Array[Double](numFftBins)
      i = 0
      while (i < numFftBins) {
        val c = spectrum(i)
        power(i) = c.re * c.re + c.im * c.im
        i += 1
      }

      val out = new Array[Float](numMelBins)
      var m = 0
      while (m < numMelBins) {
        val filter = melFilters(m)
        var energy = 0.0
        var k = 0
        while (k < filter.weights.length) {
          energy += filter.weights(k) * power(filter.startBin + k)
          k += 1
        }
        out(m) = math.log(math.max(energy, epsilon)).toFloat
        m += 1
      }
      out
    }
  }
}

object KaldiFbank {
  private def nextPowerOfTwo(n: Int): Int = {
    var p = 1
    while (p < n) p <<= 1
    p
  }

  private val instanceCache =
    new java.util.concurrent.ConcurrentHashMap[(Int, Int), KaldiFbank]()

  private def cachedInstance(sampleFrequency: Int, numMelBins: Int): KaldiFbank =
    instanceCache.computeIfAbsent(
      (sampleFrequency, numMelBins),
      _ => new KaldiFbank(sampleFrequency = sampleFrequency, numMelBins = numMelBins))

  /** Extracts fbank features, then applies per-utterance cepstral mean normalization (subtract
    * the mean of each of the `numMelBins` dimensions across time) — the standard WeSpeaker
    * inference-time normalization applied on top of raw fbank features.
    */
  def extractNormalizedFeatures(
      samples: Array[Float],
      sampleFrequency: Int = 16000,
      numMelBins: Int = 80): Array[Array[Float]] = {
    val fbank = cachedInstance(sampleFrequency, numMelBins)
    val features = fbank.extractFeatures(samples)
    if (features.isEmpty) return features

    val numFrames = features.length
    val dim = features(0).length
    val means = new Array[Double](dim)
    features.foreach(frame => {
      var d = 0
      while (d < dim) {
        means(d) += frame(d)
        d += 1
      }
    })
    var d = 0
    while (d < dim) {
      means(d) /= numFrames
      d += 1
    }

    features.map { frame =>
      Array.tabulate(dim)(d => (frame(d) - means(d)).toFloat)
    }
  }
}
