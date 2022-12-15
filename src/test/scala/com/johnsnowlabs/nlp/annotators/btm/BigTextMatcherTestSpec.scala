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

package com.johnsnowlabs.nlp.annotators.btm

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.ReadAs
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

class BigTextMatcherTestSpec extends AnyFlatSpec with BigTextMatcherBehaviors {

  "An BigTextMatcher" should s"be of type $CHUNK" taggedAs FastTest in {
    val entityExtractor = new BigTextMatcherModel
    assert(entityExtractor.outputAnnotatorType == CHUNK)
  }

  "A BigTextMatcher" should "extract entities with and without sentences" taggedAs FastTest in {
    val dataset =
      DataBuilder.basicDataBuild("Hello dolore magna aliqua. Lorem ipsum dolor. sit in laborum")
    val result = AnnotatorBuilder.withFullBigTextMatcher(dataset)
    val resultNoSentence = AnnotatorBuilder.withFullBigTextMatcher(dataset, sbd = false)
    val resultNoSentenceNoCase =
      AnnotatorBuilder.withFullBigTextMatcher(dataset, sbd = false, caseSensitive = false)
    val extractedSentenced = Annotation.collect(result, "entity").flatten.toSeq
    val extractedNoSentence = Annotation.collect(resultNoSentence, "entity").flatten.toSeq
    val extractedNoSentenceNoCase =
      Annotation.collect(resultNoSentenceNoCase, "entity").flatten.toSeq

    val expectedSentenced = Seq(
      Annotation(CHUNK, 6, 24, "dolore magna aliqua", Map("sentence" -> "0", "chunk" -> "0")),
      Annotation(CHUNK, 53, 59, "laborum", Map("sentence" -> "2", "chunk" -> "1")))

    val expectedNoSentence = Seq(
      Annotation(CHUNK, 6, 24, "dolore magna aliqua", Map("sentence" -> "0", "chunk" -> "0")),
      Annotation(CHUNK, 53, 59, "laborum", Map("sentence" -> "0", "chunk" -> "1")))

    val expectedNoSentenceNoCase = Seq(
      Annotation(CHUNK, 6, 24, "dolore magna aliqua", Map("sentence" -> "0", "chunk" -> "0")),
      Annotation(CHUNK, 27, 48, "Lorem ipsum dolor. sit", Map("sentence" -> "0", "chunk" -> "1")),
      Annotation(CHUNK, 53, 59, "laborum", Map("sentence" -> "0", "chunk" -> "2")))

    assert(extractedSentenced == expectedSentenced)
    assert(extractedNoSentence == expectedNoSentence)
    assert(extractedNoSentenceNoCase == expectedNoSentenceNoCase)
  }

  "An Entity Extractor" should "search inside sentences" taggedAs FastTest in {
    val dataset = DataBuilder.basicDataBuild("Hello dolore magna. Aliqua")
    val result = AnnotatorBuilder.withFullBigTextMatcher(dataset, caseSensitive = false)
    val extracted = Annotation.collect(result, "entity").flatten.toSeq

    assert(extracted == Seq.empty[Annotation])
  }

  "A Recursive Pipeline BigTextMatcher" should "extract entities from dataset" taggedAs FastTest in {
    val data = ContentProvider.parquetData.limit(1000)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val entityExtractor = new BigTextMatcher()
      .setInputCols("sentence", "token")
      .setStoragePath("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.TEXT)
      .setOutputCol("entity")

    val finisher = new Finisher()
      .setInputCols("entity")
      .setOutputAsArray(false)
      .setAnnotationSplitSymbol("@")
      .setValueSplitSymbol("#")

    val recursivePipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentenceDetector, tokenizer, entityExtractor, finisher))

    val m = recursivePipeline.fit(data)
    m.write.overwrite().save("./tmp_bigtm")
    m.transform(data).show(1, truncate = false)
    assert(recursivePipeline.fit(data).transform(data).filter("finished_entity == ''").count > 0)
  }

  "A big text matcher pipeline" should "work fine" taggedAs FastTest in {
    val m = PipelineModel.load("./tmp_bigtm")
    val dataset = DataBuilder.basicDataBuild("Hello dolore magna. Aliqua")
    m.transform(dataset).show(1, truncate = false)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullBigTextMatcher(
    latinBodyData)

  "A BigTextMatcher" should "also match substrings of entities" taggedAs FastTest in {

    val data =
      DataBuilder.basicDataBuild("patient has Lung Cancer", "patient Lung and Kidney Cancer")

    val tokenizedData = AnnotatorBuilder.withTokenizer(data, sbd = false)

    val entityExtractor = new BigTextMatcher()
      .setInputCols("document", "token")
      .setStoragePath("src/test/resources/entity-extractor/test-overlapping.txt", ReadAs.TEXT)
      .setOutputCol("entity")
      .setCaseSensitive(false)

    val results = entityExtractor.fit(tokenizedData).transform(tokenizedData)

    val collected = Annotation.collect(results, "entity").flatten

    val expected = Seq(
      Annotation(
        CHUNK,
        begin = 12,
        end = 15,
        result = "Lung",
        Map("sentence" -> "0", "chunk" -> "0")),
      Annotation(
        CHUNK,
        begin = 12,
        end = 22,
        result = "Lung Cancer",
        Map("sentence" -> "0", "chunk" -> "1")),
      Annotation(
        CHUNK,
        begin = 8,
        end = 11,
        result = "Lung",
        Map("sentence" -> "0", "chunk" -> "0")),
      Annotation(
        CHUNK,
        begin = 17,
        end = 22,
        result = "Kidney",
        Map("sentence" -> "0", "chunk" -> "1")),
      Annotation(
        CHUNK,
        begin = 17,
        end = 29,
        result = "Kidney Cancer",
        Map("sentence" -> "0", "chunk" -> "2")))

    assert(expected.length == collected.length)

    expected.zip(collected).map { case (expAnno: Annotation, anno: Annotation) =>
      assert(expAnno == anno)
    }

    // Test for merged chunks
    entityExtractor.setMergeOverlapping(true)

    val resultsMerged = entityExtractor.fit(tokenizedData).transform(tokenizedData)

    val collectedMerged = Annotation.collect(resultsMerged, "entity").flatten

    val expectedMerged = Seq(
      Annotation(
        CHUNK,
        begin = 12,
        end = 22,
        result = "Lung Cancer",
        Map("sentence" -> "0", "chunk" -> "0")),
      Annotation(
        CHUNK,
        begin = 8,
        end = 11,
        result = "Lung",
        Map("sentence" -> "0", "chunk" -> "0")),
      Annotation(
        CHUNK,
        begin = 17,
        end = 29,
        result = "Kidney Cancer",
        Map("sentence" -> "0", "chunk" -> "1")))

    assert(expectedMerged.length == collectedMerged.length)

    expectedMerged.zip(collectedMerged).map { case (expAnno: Annotation, anno: Annotation) =>
      assert(expAnno == anno)
    }

  }

}
