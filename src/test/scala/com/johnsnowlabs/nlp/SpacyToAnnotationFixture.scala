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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}

object SpacyToAnnotationFixture {

  val expectedMultiDocuments: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(
        DOCUMENT,
        0,
        55,
        "John went to the store last night. He bought some bread.",
        Map())),
    Seq(Annotation(DOCUMENT, 0, 47, "Hello world! How are you today? I'm fine thanks.", Map())))

  val expectedMultiSentences: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(DOCUMENT, 0, 33, "John went to the store last night.", Map("sentence" -> "0")),
      Annotation(DOCUMENT, 35, 55, "He bought some bread.", Map("sentence" -> "1"))),
    Seq(
      Annotation(DOCUMENT, 0, 11, "Hello world!", Map("sentence" -> "0")),
      Annotation(DOCUMENT, 13, 30, "How are you today?", Map("sentence" -> "1")),
      Annotation(DOCUMENT, 32, 47, "I'm fine thanks.", Map("sentence" -> "2"))))

  val expectedMultiTokens: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(TOKEN, 0, 3, "John", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 8, "went", Map("sentence" -> "0")),
      Annotation(TOKEN, 10, 11, "to", Map("sentence" -> "0")),
      Annotation(TOKEN, 13, 15, "the", Map("sentence" -> "0")),
      Annotation(TOKEN, 17, 21, "store", Map("sentence" -> "0")),
      Annotation(TOKEN, 23, 26, "last", Map("sentence" -> "0")),
      Annotation(TOKEN, 28, 32, "night", Map("sentence" -> "0")),
      Annotation(TOKEN, 33, 33, ".", Map("sentence" -> "0")),
      Annotation(TOKEN, 35, 36, "He", Map("sentence" -> "1")),
      Annotation(TOKEN, 38, 43, "bought", Map("sentence" -> "1")),
      Annotation(TOKEN, 45, 48, "some", Map("sentence" -> "1")),
      Annotation(TOKEN, 50, 54, "bread", Map("sentence" -> "1")),
      Annotation(TOKEN, 55, 55, ".", Map("sentence" -> "1"))),
    Seq(
      Annotation(TOKEN, 0, 4, "Hello", Map("sentence" -> "0")),
      Annotation(TOKEN, 6, 10, "world", Map("sentence" -> "0")),
      Annotation(TOKEN, 11, 11, "!", Map("sentence" -> "0")),
      Annotation(TOKEN, 13, 15, "How", Map("sentence" -> "1")),
      Annotation(TOKEN, 17, 19, "are", Map("sentence" -> "1")),
      Annotation(TOKEN, 21, 23, "you", Map("sentence" -> "1")),
      Annotation(TOKEN, 25, 29, "today", Map("sentence" -> "1")),
      Annotation(TOKEN, 30, 30, "?", Map("sentence" -> "1")),
      Annotation(TOKEN, 32, 32, "I", Map("sentence" -> "2")),
      Annotation(TOKEN, 33, 34, "'m", Map("sentence" -> "2")),
      Annotation(TOKEN, 36, 39, "fine", Map("sentence" -> "2")),
      Annotation(TOKEN, 41, 46, "thanks", Map("sentence" -> "2")),
      Annotation(TOKEN, 47, 47, ".", Map("sentence" -> "2"))))

  val expectedDocument: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(
        DOCUMENT,
        0,
        55,
        "John went to the store last night. He bought some bread.",
        Map())))

  val expectedTokens: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(TOKEN, 0, 3, "John", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 8, "went", Map("sentence" -> "0")),
      Annotation(TOKEN, 10, 11, "to", Map("sentence" -> "0")),
      Annotation(TOKEN, 13, 15, "the", Map("sentence" -> "0")),
      Annotation(TOKEN, 17, 21, "store", Map("sentence" -> "0")),
      Annotation(TOKEN, 23, 26, "last", Map("sentence" -> "0")),
      Annotation(TOKEN, 28, 32, "night", Map("sentence" -> "0")),
      Annotation(TOKEN, 33, 33, ".", Map("sentence" -> "0")),
      Annotation(TOKEN, 35, 36, "He", Map("sentence" -> "0")),
      Annotation(TOKEN, 38, 43, "bought", Map("sentence" -> "0")),
      Annotation(TOKEN, 45, 48, "some", Map("sentence" -> "0")),
      Annotation(TOKEN, 50, 54, "bread", Map("sentence" -> "0")),
      Annotation(TOKEN, 55, 55, ".", Map("sentence" -> "0"))))

}
