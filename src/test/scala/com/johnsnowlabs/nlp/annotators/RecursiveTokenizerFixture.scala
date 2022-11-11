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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN

object RecursiveTokenizerFixture {

  val text1 = "How can players successfully enjoy gaming at their homes?"
  val expectedTokens: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(TOKEN, 0, 2, "How", Map("sentence" -> "0")),
      Annotation(TOKEN, 4, 6, "can", Map("sentence" -> "0")),
      Annotation(TOKEN, 8, 14, "players", Map("sentence" -> "0")),
      Annotation(TOKEN, 16, 27, "successfully", Map("sentence" -> "0")),
      Annotation(TOKEN, 29, 33, "enjoy", Map("sentence" -> "0")),
      Annotation(TOKEN, 35, 40, "gaming", Map("sentence" -> "0")),
      Annotation(TOKEN, 42, 43, "at", Map("sentence" -> "0")),
      Annotation(TOKEN, 45, 49, "their", Map("sentence" -> "0")),
      Annotation(TOKEN, 51, 55, "homes", Map("sentence" -> "0")),
      Annotation(TOKEN, 56, 56, "?", Map("sentence" -> "0"))))

  val text2 = "What did you say? I'm happy to leave"
  val expectedTokens2: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(TOKEN, 0, 3, "What", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 7, "did", Map("sentence" -> "0")),
      Annotation(TOKEN, 9, 11, "you", Map("sentence" -> "0")),
      Annotation(TOKEN, 13, 15, "say", Map("sentence" -> "0")),
      Annotation(TOKEN, 16, 16, "?", Map("sentence" -> "0")),
      Annotation(TOKEN, 18, 20, "I'm", Map("sentence" -> "1")),
      Annotation(TOKEN, 22, 26, "happy", Map("sentence" -> "1")),
      Annotation(TOKEN, 28, 29, "to", Map("sentence" -> "1")),
      Annotation(TOKEN, 31, 35, "leave", Map("sentence" -> "1"))))

}
