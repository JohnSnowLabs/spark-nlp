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

import com.johnsnowlabs.nlp.SparkAccessor.spark.implicits._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

trait NormalizerBehaviors { this: AnyFlatSpec =>

  def fullNormalizerPipeline(dataset: => Dataset[Row]) {
    "A Normalizer Annotator" should "successfully transform data" taggedAs FastTest in {
      AnnotatorBuilder
        .withFullNormalizer(dataset)
        .collect()
        .foreach { row =>
          row
            .getSeq[Row](4)
            .map(Annotation(_))
            .foreach {
              case stem: Annotation if stem.annotatorType == AnnotatorType.TOKEN =>
                assert(stem.result.nonEmpty, "Annotation result exists")
              case _ =>
            }
        }
    }
  }

  def lowercasingNormalizerPipeline(dataset: => Dataset[Row]) {
    "A case-sensitive Normalizer Annotator" should "successfully transform data" taggedAs FastTest in {
      AnnotatorBuilder
        .withCaseSensitiveNormalizer(dataset)
        .collect()
        .foreach { row =>
          val tokens = row
            .getSeq[Row](3)
            .map(Annotation(_))
            .filterNot(a => a.result == "." || a.result == ",")
          val normalizedAnnotations = row.getSeq[Row](4).map(Annotation(_))
          normalizedAnnotations.foreach {
            case nToken: Annotation if nToken.annotatorType == AnnotatorType.TOKEN =>
              assert(nToken.result.nonEmpty, "Annotation result exists")
            case _ =>
          }
          normalizedAnnotations.zip(tokens).foreach {
            case (nToken: Annotation, token: Annotation) =>
              assert(nToken.result == token.result.replaceAll("[^a-zA-Z]", ""))
          }
        }
    }
  }

  def testCorrectSlangs(dataset: Dataset[Row]): Unit = {
    s"normalizer with slang dictionary " should s"successfully correct several abbreviations" taggedAs FastTest in {

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val normalizer = new Normalizer()
        .setInputCols(Array("token"))
        .setOutputCol("normalized")
        .setLowercase(true)
        .setSlangDictionary(
          ExternalResource(
            "src/test/resources/spell/slangs.txt",
            ReadAs.TEXT,
            Map("delimiter" -> ",")))

      val finisher = new Finisher()
        .setInputCols("normalized")
        .setOutputAsArray(false)
        .setIncludeMetadata(false)
        .setAnnotationSplitSymbol("@")
        .setValueSplitSymbol("#")

      val pipeline = new Pipeline()
        .setStages(Array(documentAssembler, tokenizer, normalizer, finisher))

      val model = pipeline.fit(DataBuilder.basicDataBuild("dummy"))
      val transform = model.transform(dataset)
      val normalizedWords = transform
        .select("normalized_gt", "finished_normalized")
        .map(r => (r.getString(0), r.getString(1)))
        .collect
        .toSeq

      normalizedWords.foreach(words => {
        assert(words._1 == words._2)
      })
    }
  }

  def testMultipleRegexPatterns(dataset: Dataset[Row]): Unit = {
    s"normalizer with multiple regex patterns " should s"successfully correct several words" taggedAs FastTest in {

      val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

      val tokenizer = new Tokenizer()
        .setInputCols(Array("document"))
        .setOutputCol("token")

      val normalizer = new Normalizer()
        .setInputCols(Array("token"))
        .setOutputCol("normalized")
        .setLowercase(false)
        .setCleanupPatterns(Array("[^\\pL+]", "[^a-z]"))

      val finisher = new Finisher()
        .setInputCols("normalized")
        .setOutputAsArray(false)
        .setIncludeMetadata(false)
        .setAnnotationSplitSymbol("@")
        .setValueSplitSymbol("#")

      val pipeline = new Pipeline()
        .setStages(Array(documentAssembler, tokenizer, normalizer, finisher))

      val model = pipeline.fit(DataBuilder.basicDataBuild("dummy"))
      val transform = model.transform(dataset)
      val normalizedWords = transform
        .select("normalized_gt", "finished_normalized")
        .map(r => (r.getString(0), r.getString(1)))
        .collect
        .toSeq

      normalizedWords.foreach(words => {
        assert(words._1 == words._2)
      })
    }
  }

}
