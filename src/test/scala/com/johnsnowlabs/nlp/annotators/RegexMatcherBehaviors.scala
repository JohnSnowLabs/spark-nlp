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

import com.johnsnowlabs.nlp.AnnotatorType.CHUNK
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

import scala.language.reflectiveCalls

trait RegexMatcherBehaviors { this: AnyFlatSpec =>
  def fixture(dataset: Dataset[Row], rules: Array[(String, String)], strategy: String) = new {
    val annotationDataset: Dataset[_] = AnnotatorBuilder.withRegexMatcher(dataset, strategy)
    val regexAnnotations: Array[Annotation] = annotationDataset
      .select("regex")
      .collect
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }

  }

  def customizedRulesRegexMatcher(
      dataset: => Dataset[Row],
      rules: Array[(String, String)],
      strategy: String): Unit = {
//    "A RegexMatcher Annotator with custom rules" should s"successfuly match ${rules.map(_._1).mkString(",")}" in {
//      val f = fixture(dataset, rules, strategy)
//      f.regexAnnotations.foreach { a =>
//        assert(Seq("followed by 'the'", "ceremony").contains(a.metadata("identifier")))
//      }
//    }
//
//    it should "create annotations" in {
//      val f = fixture(dataset, rules, strategy)
//      assert(f.regexAnnotations.nonEmpty)
//    }
//
//    it should "create annotations with the correct tag" in {
//      val f = fixture(dataset, rules, strategy)
//      f.regexAnnotations.foreach { a =>
//        assert(a.annotatorType == CHUNK)
//      }
//    }

    import ResourceHelper.spark.implicits._

    val sampleDataset = ResourceHelper.spark
      .createDataFrame(Seq((
        1,
        "My first sentence with the first rule. This is my second sentence with ceremonies rule.")))
      .toDF("id", "text")

    val expectedChunks = Array(
      Annotation(
        CHUNK,
        23,
        31,
        "the first",
        Map("sentence" -> "0", "chunk" -> "0", "identifier" -> "followed by 'the'")),
      Annotation(
        CHUNK,
        71,
        80,
        "ceremonies",
        Map("sentence" -> "1", "chunk" -> "0", "identifier" -> "ceremony")))

    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val sentence = new SentenceDetector().setInputCols("document").setOutputCol("sentence")

    val rulesFile = "src/test/resources/regex-matcher/rules.txt"
    val externalResource = ExternalResource(rulesFile, ReadAs.TEXT, Map("delimiter" -> ","))

    it should "respect begin and end based on each sentence" taggedAs FastTest in {
      val regexMatcher = new RegexMatcher()
        .setExternalRules(externalResource)
        .setInputCols(Array("sentence"))
        .setOutputCol("regex")
        .setStrategy(strategy)

      val pipeline = new Pipeline().setStages(Array(documentAssembler, sentence, regexMatcher))

      val results = pipeline.fit(sampleDataset).transform(sampleDataset)

      val regexChunks = results
        .select("regex")
        .as[Seq[Annotation]]
        .collect
        .flatMap(_.toSeq)
        .toSeq
        .toArray

      assert(regexChunks sameElements expectedChunks)
    }

    it should "be able to accept rules as string arrays" taggedAs FastTest in {
      val delimiter = ","
      val ruleStrings = rules.map(t => t._1 + "," + t._2)

      val regexMatcher = new RegexMatcher()
        .setRules(ruleStrings)
        .setDelimiter(delimiter)
        .setInputCols(Array("sentence"))
        .setOutputCol("regex")
        .setStrategy(strategy)

      val pipeline = new Pipeline().setStages(Array(documentAssembler, sentence, regexMatcher))

      val results = pipeline.fit(sampleDataset).transform(sampleDataset)

      val regexChunks = results
        .select("regex")
        .as[Seq[Annotation]]
        .collect
        .flatMap(_.toSeq)
        .toSeq
        .toArray

      assert(regexChunks sameElements expectedChunks)
    }

    it should "throw exceptions for externalRules and rules conflicts" taggedAs FastTest in {
      val ruleStrings = Array("")
      intercept[IllegalArgumentException] {
        new RegexMatcher()
          .setRules(ruleStrings)
          .setExternalRules(externalResource)
          .setInputCols(Array("sentence"))
          .setOutputCol("regex")
          .setStrategy(strategy)
      }

      intercept[IllegalArgumentException] {
        new RegexMatcher()
          .setExternalRules(externalResource)
          .setRules(ruleStrings)
          .setInputCols(Array("sentence"))
          .setOutputCol("regex")
          .setStrategy(strategy)
      }
    }

    it should "throw exceptions for malformed rules" taggedAs FastTest in {
      val regexMatcher = new RegexMatcher()
        .setDelimiter(",")
        .setInputCols(Array("sentence"))
        .setOutputCol("regex")
        .setStrategy(strategy)

      intercept[IllegalArgumentException] {
        regexMatcher
          .setRules(Array("1,2,3"))
          .train(dataset, None)
      }
      intercept[IllegalArgumentException] {
        regexMatcher
          .setRules(Array("1"))
          .train(dataset, None)
      }
    }

  }
}
