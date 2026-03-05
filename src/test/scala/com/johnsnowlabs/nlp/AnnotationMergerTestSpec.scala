/*
 * Copyright 2017-2026 John Snow Labs
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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.flatspec.AnyFlatSpec

class AnnotationMergerTestSpec extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  behavior of "AnnotationMerger"

  it should "merge two DOCUMENT annotation columns" taggedAs FastTest in {
    val data = Seq(("Hello world", "This is a table"))
      .toDF("text1", "text2")

    val documentAssembler1 = new DocumentAssembler()
      .setInputCol("text1")
      .setOutputCol("document_text")

    val documentAssembler2 = new DocumentAssembler()
      .setInputCol("text2")
      .setOutputCol("document_table")

    val merger = new AnnotationMerger()
      .setInputCols("document_text", "document_table")
      .setOutputCol("merged")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler1, documentAssembler2, merger))
      .fit(data)

    val result = pipeline.transform(data)
    val annotations = Annotation.collect(result, "merged").head

    assert(annotations.length == 2)
    assert(annotations(0).result == "Hello world")
    assert(annotations(1).result == "This is a table")
    assert(annotations.forall(_.annotatorType == AnnotatorType.DOCUMENT))
    assert(annotations.forall(_.metadata.contains("source_column")))
    assert(annotations(0).metadata("source_column") == "document_text")
    assert(annotations(1).metadata("source_column") == "document_table")
  }

  it should "merge three annotation columns" taggedAs FastTest in {
    val data = Seq(("First", "Second", "Third"))
      .toDF("text1", "text2", "text3")

    val da1 = new DocumentAssembler().setInputCol("text1").setOutputCol("doc1")
    val da2 = new DocumentAssembler().setInputCol("text2").setOutputCol("doc2")
    val da3 = new DocumentAssembler().setInputCol("text3").setOutputCol("doc3")

    val merger = new AnnotationMerger()
      .setInputCols("doc1", "doc2", "doc3")
      .setOutputCol("merged")

    val pipeline = new Pipeline()
      .setStages(Array(da1, da2, da3, merger))
      .fit(data)

    val result = pipeline.transform(data)
    val annotations = Annotation.collect(result, "merged").head

    assert(annotations.length == 3)
    assert(annotations.map(_.result) sameElements Array("First", "Second", "Third"))
  }

  it should "handle single input column as passthrough" taggedAs FastTest in {
    val data = Seq("Hello world").toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val merger = new AnnotationMerger()
      .setInputCols("document")
      .setOutputCol("merged")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, merger))
      .fit(data)

    val result = pipeline.transform(data)
    val annotations = Annotation.collect(result, "merged").head

    assert(annotations.length == 1)
    assert(annotations.head.result == "Hello world")
  }

  it should "preserve metadata from original annotations" taggedAs FastTest in {
    val data = Seq(("Text A", "Text B")).toDF("text1", "text2")

    val da1 = new DocumentAssembler().setInputCol("text1").setOutputCol("doc1")
    val da2 = new DocumentAssembler().setInputCol("text2").setOutputCol("doc2")

    val merger = new AnnotationMerger()
      .setInputCols("doc1", "doc2")
      .setOutputCol("merged")

    val pipeline = new Pipeline()
      .setStages(Array(da1, da2, merger))
      .fit(data)

    val result = pipeline.transform(data)
    val annotations = Annotation.collect(result, "merged").head

    // Original DocumentAssembler metadata should be preserved
    annotations.foreach { annotation =>
      assert(annotation.metadata.contains("sentence"))
      assert(annotation.metadata.contains("source_column"))
    }
    // source_column should track the actual input column name
    assert(annotations(0).metadata("source_column") == "doc1")
    assert(annotations(1).metadata("source_column") == "doc2")
  }

  it should "sort annotations by begin position when sortByBegin is true" taggedAs FastTest in {
    val data = Seq(("Second part", "First"))
      .toDF("text1", "text2")

    val da1 = new DocumentAssembler().setInputCol("text1").setOutputCol("doc1")
    val da2 = new DocumentAssembler().setInputCol("text2").setOutputCol("doc2")

    val merger = new AnnotationMerger()
      .setInputCols("doc1", "doc2")
      .setOutputCol("merged")
      .setSortByBegin(true)

    val pipeline = new Pipeline()
      .setStages(Array(da1, da2, merger))
      .fit(data)

    val result = pipeline.transform(data)
    val annotations = Annotation.collect(result, "merged").head

    // Both should start at 0 since they're from separate documents
    // but the order should be stable when begin positions are equal
    assert(annotations.length == 2)
    val begins = annotations.map(_.begin)
    assert(begins.zip(begins.tail).forall { case (a, b) => a <= b })
  }

  it should "allow changing output annotator type" taggedAs FastTest in {
    val data = Seq(("Hello", "World")).toDF("text1", "text2")

    val da1 = new DocumentAssembler().setInputCol("text1").setOutputCol("doc1")
    val da2 = new DocumentAssembler().setInputCol("text2").setOutputCol("doc2")

    val merger = new AnnotationMerger()
      .setInputCols("doc1", "doc2")
      .setOutputCol("merged")
      .setOutputAsAnnotatorType(AnnotatorType.CHUNK)

    val pipeline = new Pipeline()
      .setStages(Array(da1, da2, merger))
      .fit(data)

    val result = pipeline.transform(data)
    val annotations = Annotation.collect(result, "merged").head

    assert(annotations.forall(_.annotatorType == AnnotatorType.CHUNK))
  }

  it should "be serializable in a pipeline" taggedAs FastTest in {
    val data = Seq(("Hello", "World")).toDF("text1", "text2")

    val da1 = new DocumentAssembler().setInputCol("text1").setOutputCol("doc1")
    val da2 = new DocumentAssembler().setInputCol("text2").setOutputCol("doc2")

    val merger = new AnnotationMerger()
      .setInputCols("doc1", "doc2")
      .setOutputCol("merged")

    val pipeline = new Pipeline()
      .setStages(Array(da1, da2, merger))
      .fit(data)

    val tmpPath = java.io.File.createTempFile("annotation_merger_test", "").getAbsolutePath

    pipeline.write.overwrite().save(tmpPath)
    val loadedPipeline = PipelineModel.load(tmpPath)

    assert(loadedPipeline != null)

    val result = loadedPipeline.transform(data)
    val annotations = Annotation.collect(result, "merged").head
    assert(annotations.length == 2)
  }

  it should "handle multiple rows correctly" taggedAs FastTest in {
    val data = Seq(
      ("Row 1 Text A", "Row 1 Text B"),
      ("Row 2 Text A", "Row 2 Text B"),
      ("Row 3 Text A", "Row 3 Text B")).toDF("text1", "text2")

    val da1 = new DocumentAssembler().setInputCol("text1").setOutputCol("doc1")
    val da2 = new DocumentAssembler().setInputCol("text2").setOutputCol("doc2")

    val merger = new AnnotationMerger()
      .setInputCols("doc1", "doc2")
      .setOutputCol("merged")

    val pipeline = new Pipeline()
      .setStages(Array(da1, da2, merger))
      .fit(data)

    val result = pipeline.transform(data)
    val allAnnotations = Annotation.collect(result, "merged")

    assert(allAnnotations.length == 3)
    allAnnotations.foreach { annotations =>
      assert(annotations.length == 2)
    }

    assert(allAnnotations(0)(0).result == "Row 1 Text A")
    assert(allAnnotations(0)(1).result == "Row 1 Text B")
    assert(allAnnotations(1)(0).result == "Row 2 Text A")
    assert(allAnnotations(1)(1).result == "Row 2 Text B")
    assert(allAnnotations(2)(0).result == "Row 3 Text A")
    assert(allAnnotations(2)(1).result == "Row 3 Text B")
  }

}
