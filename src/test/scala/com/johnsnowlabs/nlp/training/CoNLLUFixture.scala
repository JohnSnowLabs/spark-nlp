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
package com.johnsnowlabs.nlp.training

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}

object CoNLLUFixture {

  def setUpExpectationsForTestWithoutExplode(): (
      Array[Seq[Annotation]],
      Array[Seq[Annotation]],
      Array[Seq[Annotation]],
      Array[Seq[Annotation]]) = {
    val expectedDocuments = Array(
      Seq(
        Annotation(
          DOCUMENT,
          0,
          36,
          "What if Google Morphed Into GoogleOS?",
          Map("training" -> "true"))),
      Seq(
        Annotation(
          DOCUMENT,
          0,
          30,
          "Google is a nice search engine.",
          Map("training" -> "true"))),
      Seq(
        Annotation(
          DOCUMENT,
          0,
          37,
          "Does anybody use it for anything else?",
          Map("training" -> "true"))))
    val expectedSentences = Array(
      Seq(
        Annotation(
          DOCUMENT,
          0,
          36,
          "What if Google Morphed Into GoogleOS?",
          Map("sentence" -> "0"))),
      Seq(Annotation(DOCUMENT, 0, 30, "Google is a nice search engine.", Map("sentence" -> "0"))),
      Seq(
        Annotation(
          DOCUMENT,
          0,
          37,
          "Does anybody use it for anything else?",
          Map("sentence" -> "0"))))
    val expectedForms = Array(
      Seq(
        Annotation(TOKEN, 0, 3, "What", Map("sentence" -> "0")),
        Annotation(TOKEN, 5, 6, "if", Map("sentence" -> "0")),
        Annotation(TOKEN, 8, 13, "Google", Map("sentence" -> "0")),
        Annotation(TOKEN, 15, 21, "Morphed", Map("sentence" -> "0")),
        Annotation(TOKEN, 23, 26, "Into", Map("sentence" -> "0")),
        Annotation(TOKEN, 28, 35, "GoogleOS", Map("sentence" -> "0")),
        Annotation(TOKEN, 36, 36, "?", Map("sentence" -> "0"))),
      Seq(
        Annotation(TOKEN, 0, 5, "Google", Map("sentence" -> "0")),
        Annotation(TOKEN, 7, 8, "is", Map("sentence" -> "0")),
        Annotation(TOKEN, 10, 10, "a", Map("sentence" -> "0")),
        Annotation(TOKEN, 12, 15, "nice", Map("sentence" -> "0")),
        Annotation(TOKEN, 17, 22, "search", Map("sentence" -> "0")),
        Annotation(TOKEN, 24, 29, "engine", Map("sentence" -> "0")),
        Annotation(TOKEN, 30, 30, ".", Map("sentence" -> "0"))),
      Seq(
        Annotation(TOKEN, 0, 3, "Does", Map("sentence" -> "0")),
        Annotation(TOKEN, 5, 11, "anybody", Map("sentence" -> "0")),
        Annotation(TOKEN, 13, 15, "use", Map("sentence" -> "0")),
        Annotation(TOKEN, 17, 18, "it", Map("sentence" -> "0")),
        Annotation(TOKEN, 20, 22, "for", Map("sentence" -> "0")),
        Annotation(TOKEN, 24, 31, "anything", Map("sentence" -> "0")),
        Annotation(TOKEN, 33, 36, "else", Map("sentence" -> "0")),
        Annotation(TOKEN, 37, 37, "?", Map("sentence" -> "0"))))
    val expectedLemmas = Array(
      Seq(
        Annotation(TOKEN, 0, 3, "what", Map("sentence" -> "0")),
        Annotation(TOKEN, 5, 6, "if", Map("sentence" -> "0")),
        Annotation(TOKEN, 8, 13, "Google", Map("sentence" -> "0")),
        Annotation(TOKEN, 15, 21, "morph", Map("sentence" -> "0")),
        Annotation(TOKEN, 23, 26, "into", Map("sentence" -> "0")),
        Annotation(TOKEN, 28, 35, "GoogleOS", Map("sentence" -> "0")),
        Annotation(TOKEN, 36, 36, "?", Map("sentence" -> "0"))),
      Seq(
        Annotation(TOKEN, 0, 5, "Google", Map("sentence" -> "0")),
        Annotation(TOKEN, 7, 8, "be", Map("sentence" -> "0")),
        Annotation(TOKEN, 10, 10, "a", Map("sentence" -> "0")),
        Annotation(TOKEN, 12, 15, "nice", Map("sentence" -> "0")),
        Annotation(TOKEN, 17, 22, "search", Map("sentence" -> "0")),
        Annotation(TOKEN, 24, 29, "engine", Map("sentence" -> "0")),
        Annotation(TOKEN, 30, 30, ".", Map("sentence" -> "0"))),
      Seq(
        Annotation(TOKEN, 0, 3, "do", Map("sentence" -> "0")),
        Annotation(TOKEN, 5, 11, "anybody", Map("sentence" -> "0")),
        Annotation(TOKEN, 13, 15, "use", Map("sentence" -> "0")),
        Annotation(TOKEN, 17, 18, "it", Map("sentence" -> "0")),
        Annotation(TOKEN, 20, 22, "for", Map("sentence" -> "0")),
        Annotation(TOKEN, 24, 31, "anything", Map("sentence" -> "0")),
        Annotation(TOKEN, 33, 36, "else", Map("sentence" -> "0")),
        Annotation(TOKEN, 37, 37, "?", Map("sentence" -> "0"))))

    (expectedDocuments, expectedSentences, expectedForms, expectedLemmas)
  }

  def setUpExpectationsForTestWithExplode(): (
      Array[Seq[Annotation]],
      Array[Seq[Annotation]],
      Array[Seq[Annotation]],
      Array[Seq[Annotation]]) = {
    val expectedDocuments = Array(
      Seq(
        Annotation(
          DOCUMENT,
          0,
          38,
          "What if Google Morphed Into GoogleOS?\n\n",
          Map("training" -> "true"))),
      Seq(
        Annotation(
          DOCUMENT,
          0,
          70,
          "Google is a nice search engine.\n\nDoes anybody use it for anything else?",
          Map("training" -> "true"))))
    val expectedSentences = Array(
      Seq(
        Annotation(
          DOCUMENT,
          0,
          36,
          "What if Google Morphed Into GoogleOS?",
          Map("sentence" -> "0"))),
      Seq(
        Annotation(DOCUMENT, 0, 30, "Google is a nice search engine.", Map("sentence" -> "0")),
        Annotation(
          DOCUMENT,
          33,
          70,
          "Does anybody use it for anything else?",
          Map("sentence" -> "1"))))
    val expectedForms = Array(
      Seq(
        Annotation(TOKEN, 0, 3, "What", Map("sentence" -> "0")),
        Annotation(TOKEN, 5, 6, "if", Map("sentence" -> "0")),
        Annotation(TOKEN, 8, 13, "Google", Map("sentence" -> "0")),
        Annotation(TOKEN, 15, 21, "Morphed", Map("sentence" -> "0")),
        Annotation(TOKEN, 23, 26, "Into", Map("sentence" -> "0")),
        Annotation(TOKEN, 28, 35, "GoogleOS", Map("sentence" -> "0")),
        Annotation(TOKEN, 36, 36, "?", Map("sentence" -> "0"))),
      Seq(
        Annotation(TOKEN, 0, 5, "Google", Map("sentence" -> "0")),
        Annotation(TOKEN, 7, 8, "is", Map("sentence" -> "0")),
        Annotation(TOKEN, 10, 10, "a", Map("sentence" -> "0")),
        Annotation(TOKEN, 12, 15, "nice", Map("sentence" -> "0")),
        Annotation(TOKEN, 17, 22, "search", Map("sentence" -> "0")),
        Annotation(TOKEN, 24, 29, "engine", Map("sentence" -> "0")),
        Annotation(TOKEN, 30, 30, ".", Map("sentence" -> "0")),
        Annotation(TOKEN, 33, 36, "Does", Map("sentence" -> "1")),
        Annotation(TOKEN, 38, 44, "anybody", Map("sentence" -> "1")),
        Annotation(TOKEN, 46, 48, "use", Map("sentence" -> "1")),
        Annotation(TOKEN, 50, 51, "it", Map("sentence" -> "1")),
        Annotation(TOKEN, 53, 55, "for", Map("sentence" -> "1")),
        Annotation(TOKEN, 57, 64, "anything", Map("sentence" -> "1")),
        Annotation(TOKEN, 66, 69, "else", Map("sentence" -> "1")),
        Annotation(TOKEN, 70, 70, "?", Map("sentence" -> "1"))))
    val expectedLemmas = Array(
      Seq(
        Annotation(TOKEN, 0, 3, "what", Map("sentence" -> "0")),
        Annotation(TOKEN, 5, 6, "if", Map("sentence" -> "0")),
        Annotation(TOKEN, 8, 13, "Google", Map("sentence" -> "0")),
        Annotation(TOKEN, 15, 21, "morph", Map("sentence" -> "0")),
        Annotation(TOKEN, 23, 26, "into", Map("sentence" -> "0")),
        Annotation(TOKEN, 28, 35, "GoogleOS", Map("sentence" -> "0")),
        Annotation(TOKEN, 36, 36, "?", Map("sentence" -> "0"))),
      Seq(
        Annotation(TOKEN, 0, 5, "Google", Map("sentence" -> "0")),
        Annotation(TOKEN, 7, 8, "be", Map("sentence" -> "0")),
        Annotation(TOKEN, 10, 10, "a", Map("sentence" -> "0")),
        Annotation(TOKEN, 12, 15, "nice", Map("sentence" -> "0")),
        Annotation(TOKEN, 17, 22, "search", Map("sentence" -> "0")),
        Annotation(TOKEN, 24, 29, "engine", Map("sentence" -> "0")),
        Annotation(TOKEN, 30, 30, ".", Map("sentence" -> "0")),
        Annotation(TOKEN, 33, 36, "do", Map("sentence" -> "1")),
        Annotation(TOKEN, 38, 44, "anybody", Map("sentence" -> "1")),
        Annotation(TOKEN, 46, 48, "use", Map("sentence" -> "1")),
        Annotation(TOKEN, 50, 51, "it", Map("sentence" -> "1")),
        Annotation(TOKEN, 53, 55, "for", Map("sentence" -> "1")),
        Annotation(TOKEN, 57, 64, "anything", Map("sentence" -> "1")),
        Annotation(TOKEN, 66, 69, "else", Map("sentence" -> "1")),
        Annotation(TOKEN, 70, 70, "?", Map("sentence" -> "1"))))

    (expectedDocuments, expectedSentences, expectedForms, expectedLemmas)
  }

}
