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

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, AnnotatorType}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

import scala.language.reflectiveCalls

trait TokenizerBehaviors { this: AnyFlatSpec =>

  def fixture(dataset: => Dataset[Row]) = new {
    val df = AnnotatorBuilder.withTokenizer(AnnotatorBuilder.withTokenizer(dataset))
    val documents = df.select("document")
    val sentences = df.select("sentence")
    val tokens = df.select("token")
    val sentencesAnnotations = sentences.collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { a =>
        Annotation(
          a.getString(0),
          a.getInt(1),
          a.getInt(2),
          a.getString(3),
          a.getMap[String, String](4))
      }
    val tokensAnnotations = tokens.collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { a =>
        Annotation(
          a.getString(0),
          a.getInt(1),
          a.getInt(2),
          a.getString(3),
          a.getMap[String, String](4))
      }

    val docAnnotations = documents.collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { a =>
        Annotation(
          a.getString(0),
          a.getInt(1),
          a.getInt(2),
          a.getString(3),
          a.getMap[String, String](4))
      }

    val corpus = docAnnotations
      .map(d => d.result)
      .mkString("")
  }

  def fullTokenizerPipeline(dataset: => Dataset[Row]) {
    "A Tokenizer Annotator" should "successfully transform data" taggedAs FastTest in {
      val f = fixture(dataset)
      assert(f.tokensAnnotations.nonEmpty, "Tokenizer should add annotators")
    }

    it should "annotate using the annotatorType of token" taggedAs FastTest in {
      val f = fixture(dataset)
      assert(f.tokensAnnotations.nonEmpty, "Tokenizer should add annotators")
      f.tokensAnnotations.foreach { a =>
        assert(
          a.annotatorType == AnnotatorType.TOKEN,
          "Tokenizer annotations type should be equal to 'token'")
      }
    }

    it should "annotate with the correct word indexes" taggedAs FastTest in {
      val f = fixture(dataset)
      f.tokensAnnotations.foreach { a =>
        val token = a.result
        val sentenceToken = f.corpus.slice(a.begin, a.end + 1)
        assert(
          sentenceToken == token,
          s"Word ($sentenceToken) from sentence at (${a.begin},${a.end}) should be equal to token ($token) inside the corpus ${f.corpus}")
      }
    }
  }
}
