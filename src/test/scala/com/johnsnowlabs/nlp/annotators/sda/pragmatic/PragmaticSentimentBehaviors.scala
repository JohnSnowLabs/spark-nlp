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

package com.johnsnowlabs.nlp.annotators.sda.pragmatic

import com.johnsnowlabs.nlp.AnnotatorType.SENTIMENT
import com.johnsnowlabs.nlp.annotators.common.TokenizedSentence
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

import scala.language.reflectiveCalls

trait PragmaticSentimentBehaviors { this: AnyFlatSpec =>

  def fixture(dataset: Dataset[Row]) = new {
    val df = AnnotatorBuilder.withPragmaticSentimentDetector(dataset)
    val sdAnnotations = Annotation.collect(df, "sentiment").flatten
  }

  def isolatedSentimentDetector(tokenizedSentences: Array[TokenizedSentence], expectedScore: Double): Unit = {
    s"tagged sentences" should s"have an expected score of $expectedScore" taggedAs FastTest in {
      val pragmaticScorer = new PragmaticScorer(ResourceHelper.parseKeyValueText(ExternalResource("src/test/resources/sentiment-corpus/default-sentiment-dict.txt", ReadAs.TEXT, Map("delimiter" -> ","))))
      val result = pragmaticScorer.score(tokenizedSentences)
      assert(result == expectedScore, s"because result: $result did not match expected: $expectedScore")
    }
  }

  def sparkBasedSentimentDetector(dataset: => Dataset[Row]): Unit = {

    "A Pragmatic Sentiment Analysis Annotator" should s"create annotations" taggedAs FastTest in {
      val f = fixture(dataset)
      assert(f.sdAnnotations.size > 0)
    }

    it should "create annotations with the correct type" taggedAs FastTest in {
      val f = fixture(dataset)
      f.sdAnnotations.foreach { a =>
        assert(a.annotatorType == SENTIMENT)
      }
    }

    it should "successfully score sentences" taggedAs FastTest in {
      val f = fixture(dataset)
      f.sdAnnotations.foreach { a =>
        assert(List("positive", "negative").contains(a.result))
      }
    }
  }
}