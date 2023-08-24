package com.johnsnowlabs.nlp.annotators.audio.feature_extractor

import breeze.linalg._
import breeze.math.Complex
import breeze.numerics.{abs, log10, pow}
import breeze.signal.fourierTr
import com.johnsnowlabs.ml.util.LinAlg.implicits.ExtendedDenseMatrix

object AudioUtils {

  /** Converts Hertz to Mels.
    *
    * Uses method proposed by Slaney 1998, adapted from huggingface transformer.
    *
    * @param freq
    *   Frequency in Hertz
    * @return
    *   Frequency in Mels
    */
  def hertzToMel(freq: Double): Double = {
    val minLogHertz = 1000.0
    val minLogMel = 15.0
    val logstep = 27.0 / Math.log(6.4)
    val mels = 3.0 * freq / 200.0

    if (freq >= minLogHertz) minLogMel + Math.log(freq / minLogHertz) * logstep
    else mels
  }

  /** Converts Hertz to Mels.
    *
    * Uses method proposed by Slaney 1998, adapted from huggingface transformer.
    *
    * @param mels
    *   Frequency in Mels
    * @return
    *   Frequency in Hertz
    */
  def melToHertz(mels: Double): Double = {

    val minLogHertz = 1000.0
    val minLogMel = 15.0
    val logstep = Math.log(6.4) / 27.0
    val freq = 200.0 * mels / 3.0

    if (mels >= minLogMel)
      minLogHertz * Math.exp(logstep * (mels - minLogMel))
    else freq
  }

  /** Creates a triangular filter bank.
    *
    * Adapted from huggingface transformer.
    *
    * @param fftFreqs
    *   Discrete frequencies of the FFT bins in Hz
    * @param filterFreqs
    *   Center frequencies of the triangular filters to create, in Hz
    * @return
    */
  private def createTriangularFilterBank(
      fftFreqs: DenseVector[Double],
      filterFreqs: DenseVector[Double]): DenseMatrix[Double] = {

    // Calculate the difference f[i+1] - f[i]
    val filterDiff: DenseVector[Double] = filterFreqs(1 to -1) - filterFreqs(0 to -2)

    // Explicit Broadcasting
    val filterFreqsMatrix =
      filterFreqs.toDenseMatrix.broadcastTo((fftFreqs.length, filterFreqs.length))
    val fftFreqsMatrix =
      fftFreqs.toDenseMatrix.t.broadcastTo((fftFreqs.length, filterFreqs.length))

    val slopes = filterFreqsMatrix - fftFreqsMatrix

    val negativeSlopes: DenseMatrix[Double] = -slopes(::, 0 to -3) // discard last two rows
    val filterFirst: DenseMatrix[Double] =
      filterDiff(0 to -2).toDenseMatrix.broadcastTo(negativeSlopes)
    val downSlopes: DenseMatrix[Double] = negativeSlopes / filterFirst

    val positiveSlopes = slopes(::, 2 to -1)
    val filterLast: DenseMatrix[Double] =
      filterDiff(1 to -1).toDenseMatrix.broadcastTo(positiveSlopes)
    val upSlopes: DenseMatrix[Double] = positiveSlopes / filterLast

    val slopeMins = min(downSlopes, upSlopes)

    max(DenseMatrix.zeros[Double](slopeMins.rows, slopeMins.cols), slopeMins)
  }

  /** Creates a matrix to convert a spectrogram to the mel scale.
    *
    * Currently only the filter bank from the Auditory Toolbox for MATLAB (Slaney 1998) is
    * supported.
    *
    * Adapted from huggingface transformer.
    *
    * @param numFrequencyBins
    *   Number of the frequency bins after used for fourier transform
    * @param numMelFilters
    *   Number of mel filters to generate
    * @param minFrequency
    *   Lowest frequency in Hz
    * @param maxFrequency
    *   Highest frequency in Hz
    * @param samplingRate
    *   Sample rate of the audio
    * @return
    */
  def melFilterBank(
      numFrequencyBins: Int,
      numMelFilters: Int,
      minFrequency: Double,
      maxFrequency: Double,
      samplingRate: Int): DenseMatrix[Double] = {

    val fftFreqs = linspace(0, samplingRate / 2, numFrequencyBins)

    val melMin = hertzToMel(minFrequency)
    val melMax = hertzToMel(maxFrequency)
    val melFreqs = linspace(melMin, melMax, numMelFilters + 2)
    val filterFreqs: DenseVector[Double] = melFreqs.map(melToHertz)

    val melFilters: DenseMatrix[Double] =
      createTriangularFilterBank(fftFreqs, filterFreqs)

    // Slaney-style mel is scaled to be approx constant energy per channel
    val enorm: DenseVector[Double] =
      2.0d / (filterFreqs(2 until numMelFilters + 2) - filterFreqs(0 until numMelFilters))

    val normedMelFilters =
      melFilters *:* enorm.toDenseMatrix.broadcastTo(melFilters)

    if (max(normedMelFilters, Axis._0).t.exists(_ == 0.0d))
      println(
        "Warning: At least one mel filter has all zero values. " +
          "The value of numMelFilters might be set too high or numFrequencyBins too low.")

    normedMelFilters
  }

  /** Pads a vector with the reflection of the vector. It is mirrored on the first and last values
    * of it along each axis.
    *
    * This method is adapted from numpy. However, reflections larger than the initial vector are
    * currently not supported.
    *
    * @param vector
    *   Vector to be padded
    * @param padding
    *   Padding for the left and right side
    * @return
    *   Vector of size `padding._1 + vector.length + padding._2`
    */
  // noinspection ReplaceToWithUntil
  def padReflective(vector: DenseVector[Double], padding: (Int, Int)): DenseVector[Double] = {
    require(
      padding._1 <= vector.length - 1 && padding._2 <= vector.length - 1,
      "Reflecting past the vector itself is currently not supported. Perhaps the padding value was set too high.")
    val paddingLeft = vector(padding._1 to 1 by -1)
    val paddingRight = vector(-2 to -padding._2 - 1 by -1)
    DenseVector.vertcat(paddingLeft, vector, paddingRight)
  }

  /** Calculates a spectrogram over one waveform using the Short-Time Fourier Transform.
    *
    * We assume that the waveform has been zero padded beforehand. Currently, this method only
    * supports the mel spectrogram. It uses the breeze implementation of the fourier transform.
    *
    * Adapted from huggingface transformer.
    *
    * How this works:
    *
    *   1. The input waveform is split into frames of size `frameLength` that are partially
    *      overlapping by `frameLength - hopLength` samples.
    *   1. Each frame is multiplied by the window.
    *   1. The DFT is taken of each windowed frame.
    *   1. The resulting rows of the spectrogram are stacked to form the final matrix.
    *
    * @param waveform
    *   The input waveform to process
    * @param window
    *   The window to use for each frame
    * @param frameLength
    *   Length of each frame
    * @param hopLength
    *   Length to advance in the waveform for each overlapping step
    * @param melFilters
    *   The mel filters to apply
    * @param power
    *   Exponent to scale the spectrogram
    * @param center
    *   Whether to center the waveform and pad reflectively
    * @param onesided
    *   Whether to only return the positive DFT
    * @param melFloor
    *   Lowest value to apply for the mel spectrogram
    * @return
    *   Log Mel Spectrogram of the waveform
    */
  def calculateSpectrogram(
      waveform: DenseVector[Double],
      window: DenseVector[Double],
      frameLength: Int,
      hopLength: Int,
      melFilters: DenseMatrix[Double],
      power: Double,
      center: Boolean = true,
      onesided: Boolean = true,
      melFloor: Double = 1e-10): DenseMatrix[Double] = {

    require(window.length == frameLength, "Window must be same size as the frame")

    // Center waveform
    val processedWaveform =
      if (center) padReflective(waveform, (frameLength / 2, frameLength / 2)) else waveform

    // Split waveform into frames of size frameLength
    val fftLength = frameLength
    val numFrames: Int = 1 + ((processedWaveform.size - frameLength) / hopLength)
    val numFrequencyBins = if (onesided) (fftLength / 2) + 1 else fftLength

    // Do the Fourier Transform for each time step
    val spectrogramRows = (0 until numFrames).map { frame: Int =>
      val windowedFrame: DenseVector[Double] = {
        val timestep = frame * hopLength
        val (frameStart, frameEnd) = (timestep, timestep + frameLength)
        processedWaveform(frameStart until frameEnd) *:* window
      }

      // TODO
      // Find a way to only compute positive imaginary part without complex conjugate (second
      // half of the FT). Right now we just have to discard it.
      val fourier = fourierTr(windowedFrame).slice(0, numFrequencyBins).toDenseMatrix
      fourier
    }

    // TODO
    // If assignment to complex matrices works at some point, this can be replaced by an
    // initialized matrix and assigned in-place.
    val spectrogram: DenseMatrix[Complex] = DenseMatrix.vertcat(spectrogramRows: _*)

    val scaledSpectrogram: DenseMatrix[Double] = pow(abs(spectrogram), power).t

    val melFilteredSpectrogram = max(melFilters.t * scaledSpectrogram, melFloor)

    val logMelSpectrogram = log10(melFilteredSpectrogram)

    logMelSpectrogram
  }

  def matrixToFloatArray(matrix: DenseMatrix[Double]): Array[Array[Float]] = {
    matrix(*, ::).map { row: DenseVector[Double] =>
      row.toArray.map(_.toFloat)
    }.toArray
  }

}
