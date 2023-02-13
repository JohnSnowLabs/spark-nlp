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

import com.johnsnowlabs.nlp.SpacyToAnnotationFixture._
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class SpacyToAnnotationTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  "SequenceToAnnotation" should "transform token sequences to SparkNLP annotations" taggedAs FastTest in {
    val spacyToAnnotation = new SpacyToAnnotation()
      .setJsonInput("src/test/resources/spacy-to-annotation/multi_doc_tokens.json")
      .setOutputAnnotatorType("TOKEN")

    val annotationsDataset = spacyToAnnotation.transform(emptyDataSet)

    val actualDocuments = AssertAnnotations.getActualResult(annotationsDataset, "document")
    AssertAnnotations.assertFields(expectedMultiDocuments, actualDocuments)

    val actualSentences = AssertAnnotations.getActualResult(annotationsDataset, "sentence")
    AssertAnnotations.assertFields(expectedMultiSentences, actualSentences)

    val actualTokens = AssertAnnotations.getActualResult(annotationsDataset, "token")
    AssertAnnotations.assertFields(expectedMultiTokens, actualTokens)

  }

  it should "work when sentence_ends is not set" in {
    val spacyToAnnotation = new SpacyToAnnotation()
      .setJsonInput("src/test/resources/spacy-to-annotation/without_sentence_ends.json")
      .setOutputAnnotatorType("TOKEN")

    val annotationsDataset = spacyToAnnotation.transform(emptyDataSet)

    val actualDocuments = AssertAnnotations.getActualResult(annotationsDataset, "document")
    AssertAnnotations.assertFields(expectedDocument, actualDocuments)

    val actualTokens = AssertAnnotations.getActualResult(annotationsDataset, "token")
    AssertAnnotations.assertFields(expectedTokens, actualTokens)
  }

  it should "raise an error when using a wrong schema" in {

    val spacyToAnnotation = new SpacyToAnnotation()
      .setOutputAnnotatorType("TOKEN")

    val someDataSet = Seq("Some schema").toDS.toDF("text")

    assertThrows[IllegalArgumentException] {
      spacyToAnnotation.transform(someDataSet)
    }

  }

}
