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

package com.johnsnowlabs.nlp.annotators.multipleannotations

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler, LightPipeline, SparkAccessor}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec

class MultiAnnotationsSpec extends AnyFlatSpec {

  import SparkAccessor.spark.implicits._

  val text = "Example text"
  val data: DataFrame =
    SparkAccessor.spark.sparkContext.parallelize(Seq(text)).toDS().toDF("text")

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val documentAssembler2 = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document2")

  private val documentAssembler3 = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document3")

  private val multipleColumns = new MultiColumnApproach()
    .setInputCols("document", "document2", "document3")
    .setOutputCol("multiple_document")

  private val pipeline = new Pipeline()
    .setStages(Array(documentAssembler, documentAssembler2, documentAssembler3, multipleColumns))

  private val pipelineModel = pipeline.fit(data)

  "An multiple annotator chunks" should "transform data " taggedAs FastTest in {

    val expectedAnnotations = Array(
      Annotation(DOCUMENT, 0, text.length - 1, text, Map("sentence" -> "0")),
      Annotation(DOCUMENT, 0, text.length - 1, text, Map("sentence" -> "0")),
      Annotation(DOCUMENT, 0, text.length - 1, text, Map("sentence" -> "0")))

    val actualAnnotations =
      Annotation.collect(pipelineModel.transform(data), "multiple_document").flatten

    actualAnnotations.zipWithIndex.foreach { case (actualAnnotation, index) =>
      val expectedAnnotation = expectedAnnotations(index)
      assert(actualAnnotation == expectedAnnotation)
    }
  }

  it should "work for LightPipeline" taggedAs FastTest in {
    val text = "My document"
    val expectedResult = Map(
      "document" -> Seq(text),
      "document2" -> Seq(text),
      "document3" -> Seq(text),
      "multiple_document" -> Seq(text, text, text)
    )

    val actualResult = new LightPipeline(pipelineModel).annotate(text)

    assert(expectedResult == actualResult)
  }

  it should "annotate with LightPipeline with 2 inputs" taggedAs FastTest in {
    val text = "My document"
    val text2 = "Example text"

    val expectedResults = Array(
      Map(
      "document" -> Seq(text),
      "document2" -> Seq(text),
      "document3" -> Seq(text),
      "multiple_document" -> Seq(text, text, text)),
      Map(
        "document" -> Seq(text2),
        "document2" -> Seq(text2),
        "document3" -> Seq(text2),
        "multiple_document" -> Seq(text2, text2, text2))
    )

    val actualResults = new LightPipeline(pipelineModel).annotate(Array(text, text2))

    actualResults.zipWithIndex.foreach { case (actualResult, index) =>
      val expectedResult = expectedResults(index)
      assert(actualResult == expectedResult)
    }
  }

  it should "fullAnnotate with LightPipeline with 2 inputs" taggedAs FastTest in {
    val text = "My document"
    val expectedAnnotationText = Annotation(DOCUMENT, 0, text.length - 1, text, Map())
    val text2 = "Example text"
    val expectedAnnotationText2 = Annotation(DOCUMENT, 0, text2.length - 1, text2, Map())
    val expectedResults = Array(
      Map(
        "document" -> Seq(expectedAnnotationText),
        "document2" -> Seq(expectedAnnotationText),
        "document3" -> Seq(expectedAnnotationText),
        "multiple_document" -> Seq(expectedAnnotationText, expectedAnnotationText, expectedAnnotationText)),
      Map(
        "document" -> Seq(expectedAnnotationText2),
        "document2" -> Seq(expectedAnnotationText2),
        "document3" -> Seq(expectedAnnotationText2),
        "multiple_document" -> Seq(expectedAnnotationText2, expectedAnnotationText2, expectedAnnotationText2))
    )

    val annotationResults = new LightPipeline(pipelineModel).fullAnnotate(Array(text, text2))

    annotationResults.zipWithIndex.foreach { case (actualResult, index) =>
      val expectedResult = expectedResults(index)
      assert(actualResult == expectedResult)
    }
  }

}
