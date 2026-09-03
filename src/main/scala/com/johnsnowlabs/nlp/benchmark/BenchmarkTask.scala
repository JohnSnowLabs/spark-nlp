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

package com.johnsnowlabs.nlp.benchmark

/** The task a [[Benchmark.evaluate]] call is scoring against, selecting both the expected shape
  * of `goldData` and which comparison engine computes the score.
  */
sealed abstract class BenchmarkTask(val name: String)

object BenchmarkTask {
  case object NER extends BenchmarkTask("NER")
  case object WordSegmentation extends BenchmarkTask("WordSegmentation")
  case object POS extends BenchmarkTask("POS")
  case object Classification extends BenchmarkTask("Classification")
  case object SpellCheck extends BenchmarkTask("SpellCheck")
  case object LanguageDetection extends BenchmarkTask("LanguageDetection")
  case object ImageClassification extends BenchmarkTask("ImageClassification")
  case object DependencyParsing extends BenchmarkTask("DependencyParsing")
  case object QuestionAnswering extends BenchmarkTask("QuestionAnswering")
  case object SpeechRecognition extends BenchmarkTask("SpeechRecognition")
  case object Translation extends BenchmarkTask("Translation")
  case object Summarization extends BenchmarkTask("Summarization")

  val values: Seq[BenchmarkTask] = Seq(
    NER,
    WordSegmentation,
    POS,
    Classification,
    SpellCheck,
    LanguageDetection,
    ImageClassification,
    DependencyParsing,
    QuestionAnswering,
    SpeechRecognition,
    Translation,
    Summarization)

  def fromString(name: String): BenchmarkTask =
    values
      .find(_.name.equalsIgnoreCase(name))
      .getOrElse(throw new IllegalArgumentException(
        s"Unknown benchmark task '$name'. Supported: ${values.map(_.name).mkString(", ")}"))
}
