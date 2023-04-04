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

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.Dataset
import org.scalatest.flatspec.AnyFlatSpec

class CoNLLUTestSpec extends AnyFlatSpec {

  val conlluFile = "src/test/resources/conllu/en.test.conllu"

  "CoNLLU" should "read documents from CoNLL-U format without explode sentences" taggedAs FastTest in {

    val (expectedDocuments, expectedSentences, expectedForms, expectedLemmas) =
      CoNLLUFixture.setUpExpectationsForTestWithoutExplode()

    val conllDataSet = CoNLLU().readDataset(ResourceHelper.spark, conlluFile)

    assertCoNLLDataSet(conllDataSet, expectedDocuments, "document")
    assertCoNLLDataSet(conllDataSet, expectedSentences, "sentence")
    assertCoNLLDataSet(conllDataSet, expectedForms, "form")
    assertCoNLLDataSet(conllDataSet, expectedLemmas, "lemma")
  }

  it should "read documents from CoNLL-U format with explode sentences" taggedAs FastTest in {

    val (expectedDocuments, expectedSentences, expectedForms, expectedLemmas) =
      CoNLLUFixture.setUpExpectationsForTestWithExplode()

    val conllDataSet =
      CoNLLU(explodeSentences = false).readDataset(ResourceHelper.spark, conlluFile)

    assertCoNLLDataSet(conllDataSet, expectedDocuments, "document")
    assertCoNLLDataSet(conllDataSet, expectedSentences, "sentence")
    assertCoNLLDataSet(conllDataSet, expectedForms, "form")
    assertCoNLLDataSet(conllDataSet, expectedLemmas, "lemma")
  }

  it should "load from file://" in {
    val (expectedDocuments, expectedSentences, expectedForms, expectedLemmas) =
      CoNLLUFixture.setUpExpectationsForTestWithoutExplode()

    val conlluFilePaths: List[String] = List(s"file:///$conlluFile", s"file://$conlluFile")

    conlluFilePaths.foreach { conlluFilePath =>
      val conllDataSet = CoNLLU().readDataset(ResourceHelper.spark, conlluFilePath)

      assertCoNLLDataSet(conllDataSet, expectedDocuments, "document")
      assertCoNLLDataSet(conllDataSet, expectedSentences, "sentence")
      assertCoNLLDataSet(conllDataSet, expectedForms, "form")
      assertCoNLLDataSet(conllDataSet, expectedLemmas, "lemma")
    }
  }

  def assertCoNLLDataSet(
      conllDataSet: Dataset[_],
      expectedResult: Array[Seq[Annotation]],
      columnName: String): Unit = {
    val actualResult = AssertAnnotations.getActualResult(conllDataSet, columnName)
    assert(expectedResult.length == actualResult.length)
    AssertAnnotations.assertFields(expectedResult, actualResult)
  }

}
