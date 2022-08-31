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

import java.io.InputStream

/** Handles raw bytes of WAV.
  *
  * Code adapted from
  * https://raw.githubusercontent.com/microsoft/SynapseML/master/cognitive/src/main/scala/com/microsoft/azure/synapse/ml/cognitive/AudioStreams.scala
  *
  * @param wavStream
  *   stream of wav bits with header stripped
  */
class WavStream(val wavStream: InputStream) {

  val stream: InputStream = parseWavHeader(wavStream)


  def read(dataBuffer: Array[Byte]): Int = {
    Math.max(0, stream.read(dataBuffer, 0, dataBuffer.length))
  }

  def readAll(): Array[Byte] = {
    ???
  }

  def close(): Unit = {
    stream.close()
  }

  // region Wav File helper functions
  private def readUInt32(inputStream: InputStream) = {
    (0 until 4).foldLeft(0) { case (n, i) => n | inputStream.read << (i * 8) }
  }

  private def readUInt16(inputStream: InputStream) = {
    (0 until 2).foldLeft(0) { case (n, i) => n | inputStream.read << (i * 8) }
  }

  // noinspection ScalaStyle
  def parseWavHeader(reader: InputStream): InputStream = {
    // Tag "RIFF"
    val data = new Array[Byte](4)
    var numRead = reader.read(data, 0, 4)
    assert((numRead == 4) && (data sameElements "RIFF".getBytes), "RIFF")

    // Chunk size
    val fileLength = readUInt32(reader)

    numRead = reader.read(data, 0, 4)
    assert((numRead == 4) && (data sameElements "WAVE".getBytes), "WAVE")

    numRead = reader.read(data, 0, 4)
    assert((numRead == 4) && (data sameElements "fmt ".getBytes), "fmt ")

    val formatSize = readUInt32(reader)
    assert(formatSize >= 16, "formatSize")

    val formatTag = readUInt16(reader)
    val channels = readUInt16(reader)
    val samplesPerSec = readUInt32(reader)
    val avgBytesPerSec = readUInt32(reader)
    val blockAlign = readUInt16(reader)
    val bitsPerSample = readUInt16(reader)
    assert(formatTag == 1, "PCM") // PCM

    assert(channels == 1, "file needs to be single channel")
    assert(samplesPerSec == 16000, "file needs to have 16000 samples per second")
    assert(bitsPerSample == 16, "file needs to have 16 bits per sample")

    // Until now we have read 16 bytes in format, the rest is cbSize and is ignored
    // for now.
    if (formatSize > 16) {
      numRead = reader.read(new Array[Byte]((formatSize - 16).toInt))
      assert(numRead == (formatSize - 16), "could not skip extended format")
    }
    // Second Chunk, data
    // tag: data.
    // TODO: other chunks might be available and need to be skipped (LIST)
    // https://stackoverflow.com/a/63929383
    numRead = reader.read(data, 0, 4)
    assert((numRead == 4) && (data sameElements "data".getBytes))

    val dataLength = readUInt32(reader)
    reader
  }
}
