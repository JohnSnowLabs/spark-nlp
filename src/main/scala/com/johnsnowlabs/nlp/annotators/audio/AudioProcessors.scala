/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.nlp.{AnnotationAudio, AnnotatorType}

import java.io.{ByteArrayInputStream, InputStream}

/** Utils to check audio files and parse audio byte arrays. */
private[nlp] object AudioProcessors {

  private final val MAGIC_WAV: Array[Byte] = "RIFF".getBytes
  private final val MAGIC_FLAC: Array[Byte] = "fLaC".getBytes

  /** Processes the byte array as a WAV file.
    *
    * Reference: https://datafireball.com/2016/08/29/wav-deepdive-into-file-format/
    * @param rawBytes
    *   Raw bytes of the audio file.
    * @return
    *   AnnotationAudio
    */
  def processAsWav(rawBytes: Array[Byte]): AnnotationAudio = {
    val rawStream: InputStream = new ByteArrayInputStream(rawBytes)
    val wavFile = WavFile.readWavStream(rawStream)

    val mNumFrames = wavFile.getNumFrames.toInt
    var mSampleRate = wavFile.getSampleRate.toInt
    val mChannels = wavFile.getNumChannels

//    val totalNoOfFrames = mNumFrames
//    val frameOffset = offsetDuration * mSampleRate
//    var tobeReadFrames = readDurationInSeconds * mSampleRate

//    if (tobeReadFrames > (totalNoOfFrames - frameOffset)) tobeReadFrames = totalNoOfFrames - frameOffset

//    if (readDurationInSeconds != -1) {
//      mNumFrames = tobeReadFrames
//      wavFile.setNumFrames(mNumFrames)
//    }

//    this.setNoOfChannels(mChannels)
//    this.setNoOfFrames(mNumFrames)
//    this.setSampleRate(mSampleRate)
//
//
//    if (sampleRate != -1) mSampleRate = sampleRate

    // Read the magnitude values across both the channels and save them as part of
    // multi-dimensional array

    val buffer = Array.ofDim[Float](mChannels, mNumFrames)
    var readFrameCount: Long = 0
    val frameOffset = 0 // TODO: Might need support for this later
    readFrameCount = wavFile.readFrames(buffer, mNumFrames, frameOffset)

    if (wavFile != null) wavFile.close()

    val resultBuffer: Array[Float] =
      if (buffer.length == 1) buffer.head
      else {
        // TODO: multiple ways to handle multi channel. Currently only average.

        buffer.foldLeft(Array.ofDim[Float](mNumFrames)) {
          (currentArray: Array[Float], channelArray: Array[Float]) =>
            channelArray
              .map { value: Float => value / mChannels }
              .zip(currentArray)
              .map { case (a: Float, b: Float) =>
                a + b
              }
        }
      }

    new AnnotationAudio(
      annotatorType = AnnotatorType.AUDIO,
      result = resultBuffer,
      metadata = Map(
        "frames" -> mNumFrames.toString,
        "sampleRate" -> mSampleRate.toString,
        "channels" -> mChannels.toString))
  }

  def processAsFlac(rawBytes: Array[Byte]): AnnotationAudio = ???

  def createAnnotationAudio(rawAudio: Array[Byte]): AnnotationAudio = {
    val magicBytes: Array[Byte] = rawAudio.slice(0, 4)

    if (magicBytes sameElements MAGIC_WAV)
      processAsWav(rawAudio)
    else if (magicBytes sameElements MAGIC_FLAC)
      processAsFlac(rawAudio)
    else
      new AnnotationAudio(
        AnnotatorType.AUDIO,
        result = Array.emptyFloatArray,
        // TODO: Notify of error
        metadata = Map.empty[String, String])

  }
}
