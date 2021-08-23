/*
 * Copyright 2017-2021 John Snow Labs
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

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.ReadAs
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._


class TextMatcherTestSpec extends FlatSpec with TextMatcherBehaviors {

  "An TextMatcher" should s"be of type $CHUNK" taggedAs FastTest in {
    val entityExtractor = new TextMatcherModel
    assert(entityExtractor.outputAnnotatorType == CHUNK)
  }

  "A TextMatcher" should "extract entities with and without sentences" taggedAs SlowTest in {
    val dataset = DataBuilder.basicDataBuild("Hello dolore magna aliqua. Lorem ipsum dolor. sit in laborum")
    val result = AnnotatorBuilder.withFullTextMatcher(dataset)
    val resultNoSentence = AnnotatorBuilder.withFullTextMatcher(dataset, sbd = false)
    val resultNoSentenceNoCase = AnnotatorBuilder.withFullTextMatcher(dataset, sbd = false, caseSensitive = false)
    val extractedSentenced = Annotation.collect(result, "entity").flatten.toSeq
    val extractedNoSentence = Annotation.collect(resultNoSentence, "entity").flatten.toSeq
    val extractedNoSentenceNoCase = Annotation.collect(resultNoSentenceNoCase, "entity").flatten.toSeq

    val expectedSentenced = Seq(
      Annotation(CHUNK, 6, 24, "dolore magna aliqua", Map("entity"->"entity", "sentence" -> "0", "chunk" -> "0")),
      Annotation(CHUNK, 53, 59, "laborum", Map("entity"->"entity", "sentence" -> "2", "chunk" -> "1"))
    )

    val expectedNoSentence = Seq(
      Annotation(CHUNK, 6, 24, "dolore magna aliqua", Map("entity"->"entity", "sentence" -> "0", "chunk" -> "0")),
      Annotation(CHUNK, 53, 59, "laborum", Map("entity"->"entity", "sentence" -> "0", "chunk" -> "1"))
    )

    val expectedNoSentenceNoCase = Seq(
      Annotation(CHUNK, 6, 24, "dolore magna aliqua", Map("entity"->"entity", "sentence" -> "0", "chunk" -> "0")),
      Annotation(CHUNK, 27, 48, "Lorem ipsum dolor. sit", Map("entity"->"entity", "sentence" -> "0", "chunk" -> "1")),
      Annotation(CHUNK, 53, 59, "laborum", Map("entity"->"entity", "sentence" -> "0", "chunk" -> "2"))
    )

    assert(extractedSentenced == expectedSentenced)
    assert(extractedNoSentence == expectedNoSentence)
    assert(extractedNoSentenceNoCase == expectedNoSentenceNoCase)
  }

  "An Entity Extractor" should "search inside sentences" taggedAs SlowTest in {
    val dataset = DataBuilder.basicDataBuild("Hello dolore magna. Aliqua")
    val result = AnnotatorBuilder.withFullTextMatcher(dataset, caseSensitive = false)
    val extracted = Annotation.collect(result, "entity").flatten.toSeq

    assert(extracted == Seq.empty[Annotation])
  }

  "A Recursive Pipeline TextMatcher" should "extract entities from dataset" taggedAs FastTest in {
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

    val entityExtractor = new TextMatcher()
      .setInputCols("sentence", "token")
      .setEntities("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.TEXT)
      .setOutputCol("entity")

    val finisher = new Finisher()
      .setInputCols("entity")
      .setOutputAsArray(false)
      .setAnnotationSplitSymbol("@")
      .setValueSplitSymbol("#")

    val recursivePipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        entityExtractor,
        finisher
      ))

    assert(recursivePipeline.fit(data).transform(data).filter("finished_entity == ''").count > 0)
  }

  "A Recursive Pipeline TextMatcher" should "extract entities from dataset without context chars" taggedAs FastTest in {
    val data = SparkAccessor.spark.createDataFrame(Seq(("LEFT PROSTATE (CORE blah BIOPSIES) jeje",""))).toDF("text","none")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")
      .setContextChars(Array(".", ",", ";", ":", "!", "?", "*", "-", "\"", "'","(",")","+","-"))
      .setSplitChars(Array("'","\"",",", "/"," ",".","|","@","#","%","&","\\$","\\[","\\]","\\(","\\)","-",";"))

    val entityExtractor = new TextMatcher()
      .setInputCols("sentence", "token")
      .setEntities("src/test/resources/entity-extractor/test-phrases-par.txt", ReadAs.TEXT)
      .setOutputCol("entity")
      .setCaseSensitive(false)
      .setBuildFromTokens(true) // Remove this line if you want to reproduce the bug

    val finisher = new Finisher()
      .setInputCols("entity")
      .setOutputAsArray(false)
      .setAnnotationSplitSymbol("@")
      .setValueSplitSymbol("#")

    val recursivePipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        entityExtractor,
        finisher
      ))

    assert(recursivePipeline.fit(data).transform(data).filter("finished_entity == 'CORE blah BIOPSIES'").count > 0)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullTextMatcher(latinBodyData)

}
