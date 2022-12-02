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
import com.johnsnowlabs.nlp.annotators.ws.{WordSegmenterApproach, WordSegmenterModel}
import com.johnsnowlabs.nlp.training.POS
import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations, SparkAccessor}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec
import org.apache.spark.sql.functions.{size, col}

import java.nio.file.{Files, Paths}

class WordSegmenterTest extends AnyFlatSpec with SparkSessionTest {

  import SparkAccessor.spark.implicits._

  private val wordSegmenter = new WordSegmenterApproach()
    .setInputCols("document")
    .setOutputCol("token")
    .setPosColumn("tags")
    .setNIterations(5)

  private def getWordSegmenterDataSet(dataSetFile: String): DataFrame = {
    val posDataSet = POS().readDataset(spark, dataSetFile)
    posDataSet

  }

  "A Word Segmenter" should "tokenize Chinese text" taggedAs FastTest in {
    val trainingDataSet =
      getWordSegmenterDataSet("src/test/resources/word-segmenter/chinese_train.utf8")
    val testDataSet = Seq("十四不是四十").toDS.toDF("text")
    val expectedResult = Array(
      Seq(
        Annotation(TOKEN, 0, 1, "十四", Map("sentence" -> "0")),
        Annotation(TOKEN, 2, 3, "不是", Map("sentence" -> "0")),
        Annotation(TOKEN, 4, 5, "四十", Map("sentence" -> "0"))))
    val pipeline = new Pipeline().setStages(Array(documentAssembler, wordSegmenter))
    val pipelineModel = pipeline.fit(trainingDataSet)
    val wsDataSet = pipelineModel.transform(testDataSet)

    val actualResult = AssertAnnotations.getActualResult(wsDataSet, "token")
    AssertAnnotations.assertFields(expectedResult, actualResult)
  }

  it should "tokenize Japanese text" taggedAs FastTest in {
    val trainingDataSet =
      getWordSegmenterDataSet("src/test/resources/word-segmenter/japanese_train.utf8")
    val testDataSet = Seq("音楽数学生理学").toDS.toDF("text")
    val expectedResult = Array(
      Seq(
        Annotation(TOKEN, 0, 2, "音楽数", Map("sentence" -> "0")),
        Annotation(TOKEN, 3, 3, "学", Map("sentence" -> "0")),
        Annotation(TOKEN, 4, 4, "生", Map("sentence" -> "0")),
        Annotation(TOKEN, 5, 6, "理学", Map("sentence" -> "0"))))

    val pipeline = new Pipeline().setStages(Array(documentAssembler, wordSegmenter))
    val pipelineModel = pipeline.fit(trainingDataSet)
    val wsDataSet = pipelineModel.transform(testDataSet)

    val actualResult = AssertAnnotations.getActualResult(wsDataSet, "token")
    AssertAnnotations.assertFields(expectedResult, actualResult)
  }

  it should "tokenize Korean text" taggedAs FastTest in {
    val trainingDataSet =
      getWordSegmenterDataSet("src/test/resources/word-segmenter/korean_train.utf8")
    val testDataSet = Seq("피부색성언어종교").toDS.toDF("text")
    val expectedResult = Array(
      Seq(
        Annotation(TOKEN, 0, 2, "피부색", Map("sentence" -> "0")),
        Annotation(TOKEN, 3, 3, "성", Map("sentence" -> "0")),
        Annotation(TOKEN, 4, 5, "언어", Map("sentence" -> "0")),
        Annotation(TOKEN, 6, 7, "종교", Map("sentence" -> "0"))))

    val pipeline = new Pipeline().setStages(Array(documentAssembler, wordSegmenter))
    val pipelineModel = pipeline.fit(trainingDataSet)
    val wsDataSet = pipelineModel.transform(testDataSet)

    val actualResult = AssertAnnotations.getActualResult(wsDataSet, "token")
    AssertAnnotations.assertFields(expectedResult, actualResult)
  }

  it should "serialize a model" taggedAs FastTest in {
    val trainingDataSet =
      getWordSegmenterDataSet("src/test/resources/word-segmenter/chinese_train.utf8")

    wordSegmenter.fit(trainingDataSet).write.overwrite().save("./tmp_chinese_tokenizer")

    assertResult(true) {
      Files.exists(Paths.get("./tmp_chinese_tokenizer"))
    }
  }

  it should "deserialize a model" taggedAs FastTest in {
    val testDataSet = Seq("十四不是四十").toDS.toDF("text")
    val expectedResult = Array(
      Seq(
        Annotation(TOKEN, 0, 1, "十四", Map("sentence" -> "0")),
        Annotation(TOKEN, 2, 3, "不是", Map("sentence" -> "0")),
        Annotation(TOKEN, 4, 5, "四十", Map("sentence" -> "0"))))

    val wordSegmenter = WordSegmenterModel.load("./tmp_chinese_tokenizer")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, wordSegmenter))
    val pipelineModel = pipeline.fit(PipelineModels.dummyDataset)
    val wsDataSet = pipelineModel.transform(testDataSet)

    val actualResult = AssertAnnotations.getActualResult(wsDataSet, "token")
    AssertAnnotations.assertFields(expectedResult, actualResult)
  }

  it should "segment multilingual texts" taggedAs SlowTest in {
    val text =
      "oem loomma สำหรับฐานลำโพง apple homepod อุปกรณ์เครื่องเสียงยึดขาตั้งไม้แข็งตั้งพื้น speaker stands null. oem loomma สำหรับฐานลำโพง apple"
    val testDataSet = Seq(text).toDS.toDF("text")
    val wordSegmenter = WordSegmenterModel
      .pretrained("wordseg_best", "th")
      .setInputCols("sentence")
      .setOutputCol("token")
      .setEnableRegexTokenizer(true)
    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, sentenceDetector, wordSegmenter))

    val resultDataSet = pipeline.fit(testDataSet).transform(testDataSet)

    val totalTokens =
      resultDataSet.select(size(col("token.result"))).collect().map(_.getAs[Int](0))
    assert(totalTokens.head > text.split(" ").length)
  }

}
