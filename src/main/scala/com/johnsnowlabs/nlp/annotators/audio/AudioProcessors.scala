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

  def processAsWav(rawBytes: Array[Byte]): AnnotationAudio = {
    val rawStream: InputStream = new ByteArrayInputStream(rawBytes)
    val wavStream = new WavStream(rawStream)
    val data = wavStream.readAll()
    ???
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
