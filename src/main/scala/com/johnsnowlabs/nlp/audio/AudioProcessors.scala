package com.johnsnowlabs.nlp.audio

import com.johnsnowlabs.nlp.AnnotatorType

import java.io.{ByteArrayInputStream, InputStream}

/** Utils to check audio files and parse audio byte arrays. */
private[audio] object AudioProcessors {

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
