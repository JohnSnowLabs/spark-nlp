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

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{
  Annotation,
  AssertAnnotations,
  DataBuilder,
  DocumentAssembler,
  SparkAccessor
}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.flatspec.AnyFlatSpec

class RegexTokenizerTestSpec extends AnyFlatSpec {

  private def assertIndexAlignment(
      tokens: Seq[Annotation],
      text: String,
      lowerCase: Boolean = false,
      removeWhitespace: Boolean = false): Unit = {
    tokens.foreach { case Annotation(_, start, end, result, _, _) =>
      var expected = text.substring(start, end + 1)
      var annotatedText = result

      if (lowerCase) {
        expected = expected.toLowerCase
        annotatedText = annotatedText.toLowerCase
      }

      if (removeWhitespace) {
        expected = expected.replace(" ", "")
        annotatedText = annotatedText.replace(" ", "")
      }

      assert(expected == annotatedText)
    }
  }

  "RegexTokenizer" should "correctly tokenize by space" taggedAs FastTest in {

    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (1, "This is my first sentence. This is my second."),
          (2, "This is my third sentence. This is my forth.")))
      .toDF("id", "text")

    val expectedTokens = Seq(
      Annotation(TOKEN, 0, 3, "this", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
      Annotation(TOKEN, 8, 9, "my", Map("sentence" -> "0")),
      Annotation(TOKEN, 11, 15, "first", Map("sentence" -> "0")),
      Annotation(TOKEN, 17, 25, "sentence.", Map("sentence" -> "0")),
      Annotation(TOKEN, 27, 30, "this", Map("sentence" -> "1")),
      Annotation(TOKEN, 32, 33, "is", Map("sentence" -> "1")),
      Annotation(TOKEN, 35, 36, "my", Map("sentence" -> "1")),
      Annotation(TOKEN, 38, 44, "second.", Map("sentence" -> "1")),
      Annotation(TOKEN, 0, 3, "this", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
      Annotation(TOKEN, 8, 9, "my", Map("sentence" -> "0")),
      Annotation(TOKEN, 11, 15, "third", Map("sentence" -> "0")),
      Annotation(TOKEN, 17, 25, "sentence.", Map("sentence" -> "0")),
      Annotation(TOKEN, 27, 30, "this", Map("sentence" -> "1")),
      Annotation(TOKEN, 32, 33, "is", Map("sentence" -> "1")),
      Annotation(TOKEN, 35, 36, "my", Map("sentence" -> "1")),
      Annotation(TOKEN, 38, 43, "forth.", Map("sentence" -> "1")))

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val regexTokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("regexToken")
      .setToLowercase(true)
      .setPattern("\\s+")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, regexTokenizer))

    val pipelineDF = pipeline.fit(testData).transform(testData)

    //    pipelineDF.select(size(pipelineDF("regexToken.result")).as("totalTokens")).show
    //    pipelineDF.select(pipelineDF("document")).show(false)
    //    pipelineDF.select(pipelineDF("sentence")).show(false)
    //    pipelineDF.select(pipelineDF("regexToken.result")).show(false)
    //    pipelineDF.select(pipelineDF("regexToken")).show(false)

    val regexTokensResults = Annotation.collect(pipelineDF, "regexToken").flatten.toSeq
    assert(regexTokensResults == expectedTokens)

  }

  "RegexTokenizer" should "correctly tokenize by patterns" taggedAs FastTest in {

    val text = "T1-T2 DATE**[12/24/13] 10/12, ph+ 90%"
    val testData = ResourceHelper.spark
      .createDataFrame(Seq((1, text)))
      .toDF("id", "text")

    val expectedTokens = Seq(
      Annotation(TOKEN, 0, 1, "t1", Map("sentence" -> "0")),
      Annotation(TOKEN, 3, 4, "t2", Map("sentence" -> "0")),
      Annotation(TOKEN, 6, 9, "date", Map("sentence" -> "0")),
      Annotation(TOKEN, 12, 21, "[12/24/13]", Map("sentence" -> "0")),
      Annotation(TOKEN, 23, 27, "10/12", Map("sentence" -> "0")),
      Annotation(TOKEN, 30, 32, "ph+", Map("sentence" -> "0")),
      Annotation(TOKEN, 34, 36, "90%", Map("sentence" -> "0")))

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val regexTokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("regexToken")
      .setToLowercase(true)
      .setPattern("([^a-zA-Z\\/0-9\\[\\]+%])")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, regexTokenizer))

    val pipelineDF = pipeline.fit(testData).transform(testData)

    //    pipelineDF.select(size(pipelineDF("regexToken.result")).as("totalTokens")).show
    //    pipelineDF.select(pipelineDF("document")).show(false)
    //    pipelineDF.select(pipelineDF("sentence")).show(false)
    //    pipelineDF.select(pipelineDF("regexToken.result")).show(false)
    //    pipelineDF.select(pipelineDF("regexToken")).show(false)

    val regexTokensResults = Annotation.collect(pipelineDF, "regexToken").flatten.toSeq
    assert(regexTokensResults == expectedTokens)
    assertIndexAlignment(regexTokensResults, text, lowerCase = true)
  }

  "a Tokenizer" should "should correctly tokenize a parsed doc" taggedAs FastTest in {

    val content = "1. T1-T2 DATE**[12/24/13] $1.99 () (10/12), ph+ 90%"
    val pattern = "\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])"

    val data = DataBuilder.basicDataBuild(content)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetect = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
      .setCustomBounds(Array("\n"))

    val tokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("regexToken")
      .setPattern(pattern)
      .setPositionalMask(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetect, tokenizer))

    val pipelineDF = pipeline.fit(data).transform(data)

    val expectedTokens = Seq(
      Annotation(TOKEN, 0, 0, "1", Map("sentence" -> "0")),
      Annotation(TOKEN, 1, 1, ".", Map("sentence" -> "0")),
      Annotation(TOKEN, 3, 4, "T1", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 5, "-", Map("sentence" -> "0")),
      Annotation(TOKEN, 6, 7, "T2", Map("sentence" -> "0")),
      Annotation(TOKEN, 9, 12, "DATE", Map("sentence" -> "0")),
      Annotation(TOKEN, 13, 13, "*", Map("sentence" -> "0")),
      Annotation(TOKEN, 14, 14, "*", Map("sentence" -> "0")),
      Annotation(TOKEN, 15, 15, "[", Map("sentence" -> "0")),
      Annotation(TOKEN, 16, 23, "12/24/13", Map("sentence" -> "0")),
      Annotation(TOKEN, 24, 24, "]", Map("sentence" -> "0")),
      Annotation(TOKEN, 26, 26, "$", Map("sentence" -> "0")),
      Annotation(TOKEN, 27, 27, "1", Map("sentence" -> "0")),
      Annotation(TOKEN, 28, 28, ".", Map("sentence" -> "0")),
      Annotation(TOKEN, 29, 30, "99", Map("sentence" -> "0")),
      Annotation(TOKEN, 32, 33, "()", Map("sentence" -> "0")),
      Annotation(TOKEN, 35, 41, "(10/12)", Map("sentence" -> "0")),
      Annotation(TOKEN, 42, 42, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 44, 45, "ph", Map("sentence" -> "0")),
      Annotation(TOKEN, 46, 46, "+", Map("sentence" -> "0")),
      Annotation(TOKEN, 48, 49, "90", Map("sentence" -> "0")),
      Annotation(TOKEN, 50, 50, "%", Map("sentence" -> "0")))

    val regexTokensResults = Annotation.collect(pipelineDF, "regexToken").flatten.toSeq
    assert(regexTokensResults == expectedTokens)
    assertIndexAlignment(regexTokensResults, content)
  }

  "RegexTokenizer" should "correctly be saved and loaded in a pipeline" taggedAs FastTest in {

    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
          (1, "This is my first sentence. This is my second."),
          (2, "This is my third sentence. This is my forth.")))
      .toDF("id", "text")

    val expectedTokens = Seq(
      Annotation(TOKEN, 0, 3, "this", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
      Annotation(TOKEN, 8, 9, "my", Map("sentence" -> "0")),
      Annotation(TOKEN, 11, 15, "first", Map("sentence" -> "0")),
      Annotation(TOKEN, 17, 25, "sentence.", Map("sentence" -> "0")),
      Annotation(TOKEN, 27, 30, "this", Map("sentence" -> "1")),
      Annotation(TOKEN, 32, 33, "is", Map("sentence" -> "1")),
      Annotation(TOKEN, 35, 36, "my", Map("sentence" -> "1")),
      Annotation(TOKEN, 38, 44, "second.", Map("sentence" -> "1")),
      Annotation(TOKEN, 0, 3, "this", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
      Annotation(TOKEN, 8, 9, "my", Map("sentence" -> "0")),
      Annotation(TOKEN, 11, 15, "third", Map("sentence" -> "0")),
      Annotation(TOKEN, 17, 25, "sentence.", Map("sentence" -> "0")),
      Annotation(TOKEN, 27, 30, "this", Map("sentence" -> "1")),
      Annotation(TOKEN, 32, 33, "is", Map("sentence" -> "1")),
      Annotation(TOKEN, 35, 36, "my", Map("sentence" -> "1")),
      Annotation(TOKEN, 38, 43, "forth.", Map("sentence" -> "1")))

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val regexTokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("regexToken")
      .setToLowercase(true)
      .setPattern("\\s+")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, regexTokenizer))

    val pipelineModel = pipeline.fit(testData)

    val expected = pipelineModel.transform(testData)
    val regexTokensResults = Annotation.collect(expected, "regexToken").flatten.toSeq
    assert(regexTokensResults == expectedTokens)

    val pipelinePath = "tmp_regex_tok_pipeline"
    pipelineModel.write.overwrite().save(pipelinePath)
    val expectedPersisted = PipelineModel.load(pipelinePath).transform(testData)
    val regexTokensPersistedResults =
      Annotation.collect(expectedPersisted, "regexToken").flatten.toSeq
    assert(regexTokensPersistedResults == expectedTokens)

  }

  private val textZipCodes = "AL 123456!, TX 54321-4444, AL :55555-4444, 12345-4444, 12345"

  "RegexTokenizer" should "test for zipcodes with no trimming" taggedAs FastTest in {

    val pattern =
      """^(\\s+)|(?=[\\s+\"\'\|:;<=>!?~{}*+,$)\(&%\\[\\]])|(?<=[\\s+\"\'\|:;<=>!?~{}*+,$)\(&%\\[\\]])|(?=\.$)"""

    val data = ResourceHelper.spark
      .createDataFrame(Seq((1, textZipCodes)))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val regexTokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")
      .setPattern(pattern)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, regexTokenizer))

    val pipeDF = pipeline.fit(data).transform(data).select("token")
    val annotatedTokens = Annotation.collect(pipeDF, "token").flatten.toSeq

    val expectedTokens = Seq(
      Annotation(TOKEN, 0, 8, "AL 123456", Map("sentence" -> "0")),
      Annotation(TOKEN, 9, 9, "!", Map("sentence" -> "0")),
      Annotation(TOKEN, 10, 10, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 11, 24, " TX 54321-4444", Map("sentence" -> "0")),
      Annotation(TOKEN, 25, 25, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 26, 29, " AL ", Map("sentence" -> "0")),
      Annotation(TOKEN, 30, 30, ":", Map("sentence" -> "0")),
      Annotation(TOKEN, 31, 40, "55555-4444", Map("sentence" -> "0")),
      Annotation(TOKEN, 41, 41, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 42, 52, " 12345-4444", Map("sentence" -> "0")),
      Annotation(TOKEN, 53, 53, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 54, 59, " 12345", Map("sentence" -> "0")))

    assert(annotatedTokens == expectedTokens)

    assertIndexAlignment(annotatedTokens, textZipCodes)
  }

  "RegexTokenizer" should "test for zipcodes with trimming and preserving indexes policies" taggedAs FastTest in {

    val pattern =
      """^(\\s+)|(?=[\\s+\"\'\|:;<=>!?~{}*+,$)\(&%\\[\\]])|(?<=[\\s+\"\'\|:;<=>!?~{}*+,$)\(&%\\[\\]])|(?=\.$)"""

    val data = ResourceHelper.spark
      .createDataFrame(Seq((1, textZipCodes)))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val regexTokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")
      .setPattern(pattern)
      .setPositionalMask(false)
      .setTrimWhitespace(true)
      .setPreservePosition(true)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, regexTokenizer))

    val pipeDF = pipeline.fit(data).transform(data).select("token")
    val annotatedTokens = Annotation.collect(pipeDF, "token").flatten.toSeq

    val expectedTokens = Seq(
      Annotation(TOKEN, 0, 8, "AL123456", Map("sentence" -> "0")),
      Annotation(TOKEN, 9, 9, "!", Map("sentence" -> "0")),
      Annotation(TOKEN, 10, 10, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 11, 24, "TX54321-4444", Map("sentence" -> "0")),
      Annotation(TOKEN, 25, 25, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 26, 29, "AL", Map("sentence" -> "0")),
      Annotation(TOKEN, 30, 30, ":", Map("sentence" -> "0")),
      Annotation(TOKEN, 31, 40, "55555-4444", Map("sentence" -> "0")),
      Annotation(TOKEN, 41, 41, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 42, 52, "12345-4444", Map("sentence" -> "0")),
      Annotation(TOKEN, 53, 53, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 54, 59, "12345", Map("sentence" -> "0")))

    assert(annotatedTokens == expectedTokens)
    assertIndexAlignment(annotatedTokens, textZipCodes, removeWhitespace = true)
  }

  "RegexTokenizer" should "test for zipcodes with trimming and no preserving indexes policies" taggedAs FastTest in {

    val pattern =
      """^(\\s+)|(?=[\\s+\"\'\|:;<=>!?~{}*+,$)\(&%\\[\\]])|(?<=[\\s+\"\'\|:;<=>!?~{}*+,$)\(&%\\[\\]])|(?=\.$)"""

    val data = ResourceHelper.spark
      .createDataFrame(Seq((1, textZipCodes)))
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val regexTokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")
      .setPattern(pattern)
      .setPositionalMask(false)
      .setTrimWhitespace(true)
      .setPreservePosition(false)

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, regexTokenizer))

    val pipeDF = pipeline.fit(data).transform(data).select("token")
    val annotatedTokens = Annotation.collect(pipeDF, "token").flatten.toSeq

    val expectedTokens = Seq(
      Annotation(TOKEN, 0, 8, "AL123456", Map("sentence" -> "0")),
      Annotation(TOKEN, 9, 9, "!", Map("sentence" -> "0")),
      Annotation(TOKEN, 10, 10, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 12, 24, "TX54321-4444", Map("sentence" -> "0")),
      Annotation(TOKEN, 25, 25, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 27, 28, "AL", Map("sentence" -> "0")),
      Annotation(TOKEN, 30, 30, ":", Map("sentence" -> "0")),
      Annotation(TOKEN, 31, 40, "55555-4444", Map("sentence" -> "0")),
      Annotation(TOKEN, 41, 41, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 43, 52, "12345-4444", Map("sentence" -> "0")),
      Annotation(TOKEN, 53, 53, ",", Map("sentence" -> "0")),
      Annotation(TOKEN, 55, 59, "12345", Map("sentence" -> "0")))

    assert(annotatedTokens == expectedTokens)
    assertIndexAlignment(annotatedTokens, textZipCodes, removeWhitespace = true)

  }

  "RegexTokenizer" should "output same token index regardless of positional mask" taggedAs FastTest in {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val pattern = "\\s+|(?=[-.:;*+,$&%\\[\\]\\/])|(?<=[-.:;*+,$&%\\[\\]\\/])"

    val regexTokenizerPosMaskTrue = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("regex_token_pm_true")
      .setPattern(pattern)
      .setPositionalMask(true)

    val regexTokenizerPosMaskFalse = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("regex_token_pm_false")
      .setPattern(pattern)
      .setPositionalMask(false)

    val pipeline = new Pipeline()
      .setStages(
        Array(documentAssembler, sentence, regexTokenizerPosMaskTrue, regexTokenizerPosMaskFalse))

    val sampleText = "Estrogen Receptor Positive 2-3+ 90 Favorable."

    val data = ResourceHelper.spark
      .createDataFrame(Seq((1, sampleText)))
      .toDF("id", "text")

    val pipeDF = pipeline.fit(data).transform(data)

    val annotatedTokensPmTrue = Annotation.collect(pipeDF, "regex_token_pm_true").flatten.toSeq
    val annotatedTokensPmFalse = Annotation.collect(pipeDF, "regex_token_pm_false").flatten.toSeq

    val metaMap = Map("sentence" -> "0")
    val expectedAnnotations =
      Seq(
        Annotation(TOKEN, 0, 7, "Estrogen", metaMap),
        Annotation(TOKEN, 9, 16, "Receptor", metaMap),
        Annotation(TOKEN, 18, 25, "Positive", metaMap),
        Annotation(TOKEN, 27, 27, "2", metaMap),
        Annotation(TOKEN, 28, 28, "-", metaMap),
        Annotation(TOKEN, 29, 29, "3", metaMap),
        Annotation(TOKEN, 30, 30, "+", metaMap),
        Annotation(TOKEN, 32, 33, "90", metaMap),
        Annotation(TOKEN, 35, 43, "Favorable", metaMap),
        Annotation(TOKEN, 44, 44, ".", metaMap))

    assert(
      expectedAnnotations == annotatedTokensPmTrue,
      "Token with positional mask did not match with expected token.")
    assert(
      expectedAnnotations == annotatedTokensPmFalse,
      "Token without positional mask did not match with expected token.")

    assertIndexAlignment(annotatedTokensPmTrue, sampleText)
    assertIndexAlignment(annotatedTokensPmFalse, sampleText)

  }

  "RegexTokenizer" should "produce correct indexes for matches without space" in {
    val regexTokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("regex_token_pm_true")
      .setPattern("(?=.)") // split at every character

    val text = "abcdef"
    val sentence = Seq(Sentence(text, 0, text.length - 1, 0))

    val expected: Seq[IndexedToken] = text.toSeq.zipWithIndex.map { case (t: Char, i: Int) =>
      IndexedToken(t.toString, i, i)
    }

    val result: Seq[IndexedToken] =
      regexTokenizer.tag(sentence).head.indexedTokens

    assert(expected == result)

    regexTokenizer.setPositionalMask(true)
    val resultPosMask: Seq[IndexedToken] =
      regexTokenizer.tag(sentence).head.indexedTokens

    assert(expected == resultPosMask)
  }

  "RegexTokenizer" should "return correct indexes for sentences with posMask enabled" taggedAs FastTest in {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetectorDL = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    // Explanation: Split at any whitespace OR
    //              before special characters OR
    //              after special characters
    val regex_pattern =
      """\s+|(?=[-.:;"*+,$&%\[\]\(\)\/])|(?<=[-.:;"*+,$&%\[\]\(\)\/])"""

    val regexTokenizer = new RegexTokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")
      .setPattern(regex_pattern)
      .setPositionalMask(true)

    val sampleText = "First sentence. second sentence."

    val data = ResourceHelper.spark
      .createDataFrame(Seq((1, sampleText)))
      .toDF("id", "text")

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, sentenceDetectorDL, regexTokenizer))

    val fittedPipe = pipeline.fit(data).transform(data)

    val tokenResult = Annotation.collect(fittedPipe, "token").flatten.toSeq

    assertIndexAlignment(tokenResult, sampleText)
  }

  it should "tokenize with sentence index" taggedAs FastTest in {
    import SparkAccessor.spark.implicits._

    val text = "This is a sentence. This is another one"
    val testDataSet = Seq(text).toDS.toDF("text")

    val expectedEntitiesFromText1: Array[Seq[Annotation]] = Array(
      Seq(
        Annotation(TOKEN, 0, 3, "This", Map("sentence" -> "0")),
        Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
        Annotation(TOKEN, 8, 8, "a", Map("sentence" -> "0")),
        Annotation(TOKEN, 10, 18, "sentence.", Map("sentence" -> "0"))),
      Seq(
        Annotation(TOKEN, 20, 23, "This", Map("sentence" -> "1")),
        Annotation(TOKEN, 25, 26, "is", Map("sentence" -> "1")),
        Annotation(TOKEN, 28, 34, "another", Map("sentence" -> "1")),
        Annotation(TOKEN, 36, 38, "one", Map("sentence" -> "1"))))

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setExplodeSentences(true)

    val tokenizer = new RegexTokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentenceDetector, tokenizer))

    val resultDataSet = pipeline.fit(testDataSet).transform(testDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "token")

    AssertAnnotations.assertFields(expectedEntitiesFromText1, actualEntities)
  }

}
