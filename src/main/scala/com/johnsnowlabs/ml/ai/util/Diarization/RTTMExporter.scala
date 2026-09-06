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

package com.johnsnowlabs.ml.ai.util.Diarization

import com.johnsnowlabs.nlp.Annotation

/** One `SPEAKER`-type line of an RTTM file: a speaker's turn on a given recording. */
case class RTTMSegment(uri: String, tbegSeconds: Double, tdurSeconds: Double, speaker: String) {
  def tendSeconds: Double = tbegSeconds + tdurSeconds
}

/** Converts between Spark NLP's `SPEAKER`-typed `Annotation`s and RTTM (Rich Transcription Time
  * Marked), the standard NIST/DIHARD interchange format for diarization — required both for
  * interop with external tooling and to score against ground-truth RTTM from a real corpus (see
  * `DiarizationErrorRateEngine`).
  *
  * Format (one line per turn):
  * {{{
  * SPEAKER <uri> <channel> <tbeg> <tdur> <NA> <NA> <speaker> <NA> <NA>
  * }}}
  * `tbeg`/`tdur` are in seconds. Overlapping turns (e.g. from `SpeakerDiarizer`'s overlap
  * surfacing) naturally produce overlapping RTTM lines, which is valid RTTM.
  */
object RTTMExporter {

  private val channel = "1"
  private val na = "<NA>"

  /** Renders `annotations` (expected `annotatorType == AnnotatorType.SPEAKER`, `begin`/`end` in
    * milliseconds, `metadata("speaker")` set) as RTTM text for recording `uri`.
    */
  def toRTTM(annotations: Seq[Annotation], uri: String): String = {
    annotations
      .map { a =>
        val speaker = a.metadata.getOrElse("speaker", "UNKNOWN")
        val tbeg = a.begin / 1000.0
        val tdur = math.max(0.0, a.end - a.begin) / 1000.0
        f"SPEAKER $uri $channel $tbeg%.3f $tdur%.3f $na $na $speaker $na $na"
      }
      .mkString("\n")
  }

  /** Parses RTTM text into segments. Non-`SPEAKER` lines (e.g. other RTTM record types, blank
    * lines, or `;;`-prefixed comments) are skipped rather than treated as errors, matching how
    * real-world RTTM files from different tools are commonly assembled.
    */
  def fromRTTM(text: String): Seq[RTTMSegment] = {
    text.linesIterator
      .map(_.trim)
      .filter(line => line.nonEmpty && !line.startsWith(";;"))
      .flatMap { line =>
        val fields = line.split("\\s+")
        if (fields.length >= 8 && fields(0) == "SPEAKER") {
          try {
            Some(
              RTTMSegment(
                uri = fields(1),
                tbegSeconds = fields(3).toDouble,
                tdurSeconds = fields(4).toDouble,
                speaker = fields(7)))
          } catch {
            case _: NumberFormatException => None
          }
        } else None
      }
      .toSeq
  }
}
