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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.NODE
import com.johnsnowlabs.nlp.annotators.{GraphExtraction, GraphExtractionFixture, SparkSessionTest}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Dataset
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable

class GraphFinisherTest extends AnyFlatSpec with SparkSessionTest with GraphExtractionFixture {

  import spark.implicits._

  private val textDataSet = Seq("Bruce Wayne lives in Gotham").toDS.toDF("text")

  "GraphFinisher" should "raise an error when NODE annotator type column does not exist" taggedAs FastTest in {

    val finisher = new GraphFinisher().setInputCol("token").setOutputCol("finisher")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, finisher))

    assertThrows[IllegalArgumentException] {
      pipeline.fit(textDataSet)
    }

  }

  it should "output graph in RDF format as array" taggedAs FastTest in {
    val testDataSet = getDeepEntities(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("document", "token", "entities")
      .setOutputCol("graph")
      .setRelationshipTypes(Array("sees-PER"))
      .setMergeEntities(false)
      .setExplodeEntities(false)
    val finisher = new GraphFinisher().setInputCol("graph").setOutputCol("finisher")
    val pipeline = new Pipeline().setStages(Array(graphExtractor, finisher))
    val expectedGraph = List(
      Seq(Seq("sees", "nsubj", "John")),
      Seq(Seq("sees", "ccomp", "goes"), Seq("goes", "nsubj", "Bill")),
      Seq(
        Seq("sees", "ccomp", "goes"),
        Seq("goes", "nsubj", "Bill"),
        Seq("Bill", "conj", "Mary")))

    val graphDataSet = pipeline.fit(testDataSet).transform(testDataSet)
    val actualGraph = getFinisherAsArray(graphDataSet)

    assert(actualGraph == expectedGraph)
  }

  it should "output graph in RDF format as string" taggedAs FastTest in {
    val testDataSet = getDeepEntities(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("document", "token", "entities")
      .setOutputCol("graph")
      .setRelationshipTypes(Array("sees-PER"))
      .setMergeEntities(false)
      .setExplodeEntities(false)
    val finisher =
      new GraphFinisher().setInputCol("graph").setOutputCol("finisher").setOutputAsArray(false)
    val expectedGraph = List(
      Seq(
        Seq("(sees,nsubj,John)"),
        Seq("(sees,ccomp,goes)", "(goes,nsubj,Bill)"),
        Seq("(sees,ccomp,goes)", "(goes,nsubj,Bill)", "(Bill,conj,Mary)")))

    val pipeline = new Pipeline().setStages(Array(graphExtractor, finisher))
    val graphDataSet = pipeline.fit(testDataSet).transform(testDataSet)

    val actualGraph = getFinisher(graphDataSet, "finisher")
    assert(actualGraph == expectedGraph)
  }

  it should "output metadata column when includeMetadata parameter is set to true" taggedAs FastTest in {
    val testDataSet = getDeepEntities(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("document", "token", "entities")
      .setOutputCol("graph")
      .setRelationshipTypes(Array("sees-PER"))
      .setMergeEntities(false)
      .setExplodeEntities(false)
    val finisher = new GraphFinisher()
      .setInputCol("graph")
      .setOutputCol("finisher")
      .setIncludeMetadata(true)
      .setOutputAsArray(false)
    val expectedMetadata = List(Seq("(sees,PER)"))
    val expectedGraph = List(
      Seq(
        Seq("(sees,nsubj,John)"),
        Seq("(sees,ccomp,goes)", "(goes,nsubj,Bill)"),
        Seq("(sees,ccomp,goes)", "(goes,nsubj,Bill)", "(Bill,conj,Mary)")))

    val pipeline = new Pipeline().setStages(Array(graphExtractor, finisher))
    val graphDataSet = pipeline.fit(testDataSet).transform(testDataSet)

    val actualGraph = getFinisher(graphDataSet, "finisher")
    assert(actualGraph == expectedGraph)
    val actualMetadata = getFinisher(graphDataSet, "finisher_metadata")
    assert(actualMetadata == expectedMetadata)
  }

  it should "output paths when exploding entities" taggedAs FastTest in {
    val testDataSet = getUniqueEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
      .setMergeEntities(false)
    val finisher = new GraphFinisher()
      .setInputCol("graph")
      .setOutputCol("finisher")
      .setOutputAsArray(false)
      .setIncludeMetadata(true)
    val expectedMetadata = List(Seq("(ORG,TIME)", "(ORG,LOC)", "(TIME,LOC)"))
    val expectedGraph = List(
      Seq(
        Seq("(canceled,nsubj,United)"),
        Seq("(canceled,obj,flights)", "(flights,compound,morning)"),
        Seq("(canceled,nsubj,United)"),
        Seq("(canceled,obj,flights)", "(flights,nmod,Houston)"),
        Seq("(canceled,obj,flights)", "(flights,compound,morning)"),
        Seq("(canceled,obj,flights)", "(flights,nmod,Houston)")))

    val pipeline = new Pipeline().setStages(Array(graphExtractor, finisher))
    val graphDataSet = pipeline.fit(testDataSet).transform(testDataSet)

    val actualGraph = getFinisher(graphDataSet, "finisher")
    assert(actualGraph == expectedGraph)
    val actualMetadata = getFinisher(graphDataSet, "finisher_metadata")
    assert(actualMetadata == expectedMetadata)
  }

  it should "annotate output graph" in {
    val metadata = Map(
      "relationship" -> "sees, PER",
      "path1" -> "sees,nsubj,John",
      "path2" -> "sees,ccomp,goes,nsubj,Bill",
      "path3" -> "sees,ccomp,goes,nsubj,Bill,conj,Mary")
    val graphFinisher = new GraphFinisher()
    val expectedResult = "[(sees,nsubj,John)],[(sees,ccomp,goes),(goes,nsubj,Bill)]," +
      "[(sees,ccomp,goes),(goes,nsubj,Bill),(Bill,conj,Mary)]"
    val expectedAnnotated = Seq(Annotation(NODE, 0, 0, expectedResult, Map()))

    val actualAnnotated = graphFinisher.annotate(metadata)

    assert(actualAnnotated == expectedAnnotated)
  }

  private def getFinisherAsArray(dataSet: Dataset[_]) = {
    val paths = dataSet
      .select("finisher")
      .rdd
      .map { row =>
        val result: Seq[Seq[String]] =
          row.get(0).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[String]]]
        result
      }
      .collect()
      .toList
    paths.flatten
  }

  private def getFinisher(dataSet: Dataset[_], column: String) = {
    val paths = dataSet
      .select(column)
      .rdd
      .map { row =>
        val result: Seq[String] = row.get(0).asInstanceOf[mutable.WrappedArray[String]]
        result
      }
      .collect()
      .toList
    paths
  }

}
