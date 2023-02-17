/*
 * Copyright 2017-2023 John Snow Labs
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

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.training.SpacyToAnnotationFixture._
import com.johnsnowlabs.nlp.AssertAnnotations
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class SpacyToAnnotationTest extends AnyFlatSpec with SparkSessionTest {

  val resourcesPath = "src/test/resources/spacy-to-annotation"

  "SequenceToAnnotation" should "transform token sequences to SparkNLP annotations" taggedAs FastTest in {
    val spacyToAnnotation = new SpacyToAnnotation()

    val annotationsDataset =
      spacyToAnnotation.readJsonFile(spark, s"$resourcesPath/multi_doc_tokens.json")

    val actualDocuments = AssertAnnotations.getActualResult(annotationsDataset, "document")
    AssertAnnotations.assertFields(expectedMultiDocuments, actualDocuments)

    val actualSentences = AssertAnnotations.getActualResult(annotationsDataset, "sentence")
    AssertAnnotations.assertFields(expectedMultiSentences, actualSentences)

    val actualTokens = AssertAnnotations.getActualResult(annotationsDataset, "token")
    AssertAnnotations.assertFields(expectedMultiTokens, actualTokens)

  }

  it should "work when sentence_ends is not set" in {
    val spacyToAnnotation = new SpacyToAnnotation()

    val annotationsDataset =
      spacyToAnnotation.readJsonFile(spark, s"$resourcesPath/without_sentence_ends.json")

    val actualDocuments = AssertAnnotations.getActualResult(annotationsDataset, "document")
    AssertAnnotations.assertFields(expectedDocument, actualDocuments)

    val actualTokens = AssertAnnotations.getActualResult(annotationsDataset, "token")
    AssertAnnotations.assertFields(expectedTokens, actualTokens)
  }

  it should "raise an error when using a wrong jsonFilePath" in {

    val spacyToAnnotation = new SpacyToAnnotation()

    assertThrows[IllegalArgumentException] {
      spacyToAnnotation.readJsonFile(spark, s"")
    }

  }

}
