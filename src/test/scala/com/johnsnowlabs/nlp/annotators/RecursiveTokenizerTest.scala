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

import com.johnsnowlabs.nlp.AssertAnnotations
import com.johnsnowlabs.nlp.annotators.RecursiveTokenizerFixture.{
  expectedTokens,
  expectedTokens2,
  text1,
  text2
}
import org.scalatest.flatspec.AnyFlatSpec

class RecursiveTokenizerTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  "RecursiveTokenizer" should "tokenize sentence" in {

    val textDataSet = Seq(text1).toDS.toDF("text")
    val recursiveTokenizer = new RecursiveTokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")
    pipeline.setStages(Array(documentAssembler, sentenceDetector, recursiveTokenizer))

    val resultDataSet = pipeline.fit(textDataSet).transform(textDataSet)

    val actualTokens = AssertAnnotations.getActualResult(resultDataSet, "token")
    AssertAnnotations.assertFields(expectedTokens, actualTokens)
  }

  it should "tokenize sentences" in {

    val textDataSet = Seq(text2).toDS.toDF("text")
    val recursiveTokenizer = new RecursiveTokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")
    pipeline.setStages(Array(documentAssembler, sentenceDetector, recursiveTokenizer))

    val resultDataSet = pipeline.fit(textDataSet).transform(textDataSet)

    val actualTokens = AssertAnnotations.getActualResult(resultDataSet, "token")
    AssertAnnotations.assertFields(expectedTokens2, actualTokens)
  }

}
