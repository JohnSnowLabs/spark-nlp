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

import com.johnsnowlabs.nlp.AnnotatorType.NODE
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations, GraphFinisher}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable

class GraphExtractionTest extends AnyFlatSpec with SparkSessionTest with GraphExtractionFixture {

  spark.conf.set("spark.sql.crossJoin.enabled", "true")

  "Graph Extraction" should "return dependency graphs between all entities" taggedAs FastTest in {

    val testDataSet = getUniqueEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
      .setIncludeEdges(false)
    val expectedGraph = Array(
      Seq(
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "ORG,TIME",
            "left_path" -> "canceled,United",
            "right_path" -> "canceled,flights,morning")),
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "ORG,LOC",
            "left_path" -> "canceled,United",
            "right_path" -> "canceled,flights,Houston")),
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "TIME,LOC",
            "left_path" -> "canceled,flights,morning",
            "right_path" -> "canceled,flights,Houston"))))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "return dependency graphs for a pair of entities" taggedAs FastTest in {
    val testDataSet = getUniqueEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
      .setEntityTypes(Array("ORG-LOC"))
      .setIncludeEdges(false)
    val expectedGraph = Array(
      Seq(
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "ORG,LOC",
            "left_path" -> "canceled,United",
            "right_path" -> "canceled,flights,Houston"))))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "return dependency graphs for a subset of entities" taggedAs FastTest in {
    val testDataSet = getUniqueEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
      .setEntityTypes(Array("ORG-LOC", "ORG-TIME"))
      .setIncludeEdges(false)
    val expectedGraph = Array(
      Seq(
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "ORG,LOC",
            "left_path" -> "canceled,United",
            "right_path" -> "canceled,flights,Houston")),
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "ORG,TIME",
            "left_path" -> "canceled,United",
            "right_path" -> "canceled,flights,morning"))))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "return dependency graphs when entities are ambiguous" taggedAs FastTest in {
    val testDataSet = getAmbiguousEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
      .setEntityTypes(Array("ORG-LOC"))
      .setIncludeEdges(false)
    val expectedGraph = Array(
      Seq(
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "ORG,LOC",
            "left_path" -> "canceled,United",
            "right_path" -> "canceled,flights,Houston")),
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "ORG,LOC",
            "left_path" -> "canceled,United",
            "right_path" -> "canceled,flights,Houston,Dallas"))))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)

  }

  it should "exclude 0 length paths" taggedAs FastTest in {
    val testDataSet = getAmbiguousEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
      .setEntityTypes(Array("LOC-LOC"))
      .setIncludeEdges(false)
    val expectedGraph = Array(
      Seq(
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "LOC,LOC",
            "left_path" -> "canceled,flights,Houston,Dallas",
            "right_path" -> "canceled,flights,Houston"))))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "extract graphs for each sentence" taggedAs FastTest in {
    val testDataSet = getEntitiesFromTwoSentences(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
      .setEntityTypes(Array("LOC-TIME"))
      .setIncludeEdges(false)
    val expectedGraph = Array(
      Seq(
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "LOC,TIME",
            "left_path" -> "canceled,flights,Houston",
            "right_path" -> "canceled,flights,morning")),
        Annotation(
          NODE,
          59,
          60,
          "go",
          Map(
            "entities" -> "LOC,TIME",
            "left_path" -> "go,London",
            "right_path" -> "go,tomorrow"))))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "filter a sentence when filtering whit min sentence parameter" taggedAs FastTest in {

    val testDataSet = getEntitiesFromTwoSentences(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
      .setEntityTypes(Array("LOC-TIME"))
      .setMinSentenceSize(33)
      .setIncludeEdges(false)
    val expectedGraph = Array(
      Seq(
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "LOC,TIME",
            "left_path" -> "canceled,flights,Houston",
            "right_path" -> "canceled,flights,morning"))))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "return empty NODE token when filtering all long sentence whit max sentence parameter" taggedAs FastTest in {
    val testDataSet = getUniqueEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
      .setEntityTypes(Array("ORG-LOC"))
      .setMaxSentenceSize(5)
      .setIncludeEdges(false)
    val expectedGraph = Array(Seq(Annotation(NODE, 0, 0, "", Map())))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "handle overlapping entities" taggedAs SlowTest in {
    val testDataSet = getOverlappingEntities(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setMergeEntities(true)
      .setRelationshipTypes(Array("person-PER", "person-LOC"))

    val graphDataSet = graphExtractor.transform(testDataSet)
    graphDataSet.show(false)
  }

  it should "start traversing from any node" taggedAs FastTest in {
    val testDataSet = getDeepEntities(spark, tokenizerPipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("document", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
      .setEntityTypes(Array("PER-LOC"))
      .setRootTokens(Array("goes"))
      .setIncludeEdges(false)
    val expectedGraph = Array(
      Seq(
        Annotation(
          NODE,
          51,
          54,
          "goes",
          Map(
            "entities" -> "PER,LOC",
            "left_path" -> "goes,Bill",
            "right_path" -> "goes,Pasadena"))))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "find paths between relationship types" taggedAs FastTest in {
    val testDataSet = getDeepEntities(spark, tokenizerPipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("document", "token", "entities")
      .setOutputCol("graph")
      .setRelationshipTypes(Array("sees-PER"))

    val expectedGraph = Array(
      Seq(Annotation(
        NODE,
        32,
        35,
        "sees",
        Map(
          "relationship" -> "sees,PER",
          "path1" -> "sees,nsubj,John",
          "path2" -> "sees,ccomp,goes,nsubj,Bill",
          "path3" -> "sees,ccomp,goes,nsubj,Bill,conj,Mary"))))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "output empty path when token node does not exist" taggedAs FastTest in {
    val testDataSet = getDeepEntities(spark, tokenizerPipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("document", "token", "entities")
      .setOutputCol("graph")
      .setRelationshipTypes(Array("whatever-PER"))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val result = graphDataSet
      .select("graph")
      .rdd
      .map(row => row(0).asInstanceOf[mutable.WrappedArray[String]])
      .collect()
      .toList
    assert(result.head.isEmpty)
  }

  it should "find paths between relationship types for several relationships" taggedAs FastTest in {
    val testDataSet = getDeepEntities(spark, tokenizerPipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("document", "token", "entities")
      .setOutputCol("graph")
      .setRelationshipTypes(Array("goes-PER", "goes-LOC"))
      .setIncludeEdges(false)
    val expectedGraph = Array(
      Seq(
        Annotation(
          NODE,
          51,
          54,
          "goes",
          Map("relationship" -> "goes,PER", "path1" -> "goes,Bill", "path2" -> "goes,Bill,Mary")),
        Annotation(
          NODE,
          51,
          54,
          "goes",
          Map("relationship" -> "goes,LOC", "path1" -> "goes,Pasadena"))))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "find paths with edges between relationship types when merging entities" taggedAs FastTest in {
    val testDataSet = getPubTatorEntities(spark, tokenizerPipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("document", "token", "entities")
      .setOutputCol("graph")
      .setRelationshipTypes(Array("polymorphisms-GENE", "polymorphisms-DISEASE"))
    val expectedGraph = Array(
      Seq(
        Annotation(
          NODE,
          32,
          44,
          "polymorphisms",
          Map(
            "relationship" -> "polymorphisms,GENE",
            "path1" -> "polymorphisms,nsubj,Influence,nmod,gene,amod,interleukin-6")),
        Annotation(
          NODE,
          32,
          44,
          "polymorphisms",
          Map(
            "relationship" -> "polymorphisms,DISEASE",
            "path1" -> "polymorphisms,nmod,coronary_artery_calcification"))))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "output paths with edges when exploding entities" taggedAs FastTest in {
    val testDataSet = getUniqueEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
    val expectedGraph = Array(
      Seq(
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "ORG,TIME",
            "left_path" -> "canceled,nsubj,United",
            "right_path" -> "canceled,obj,flights,compound,morning")),
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "ORG,LOC",
            "left_path" -> "canceled,nsubj,United",
            "right_path" -> "canceled,obj,flights,nmod,Houston")),
        Annotation(
          NODE,
          7,
          14,
          "canceled",
          Map(
            "entities" -> "TIME,LOC",
            "left_path" -> "canceled,obj,flights,compound,morning",
            "right_path" -> "canceled,obj,flights,nmod,Houston"))))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "output paths when Typed Dependency Parser cannot label relations" taggedAs SlowTest in {
    val testDataSet = getEntitiesWithNoTypeParserOutput(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
      .setMergeEntities(true)
      .setMergeEntitiesIOBFormat("IOB")
      .setIncludeEdges(false)
    val expectedGraph = Array(
      Seq(
        Annotation(
          NODE,
          15,
          20,
          "taking",
          Map(
            "entities" -> "Medication,Diagnosis",
            "left_path" -> "taking,pills,paracetamol",
            "right_path" -> "taking,disease,due,to,heart")),
        Annotation(
          NODE,
          15,
          20,
          "taking",
          Map(
            "entities" -> "Medication,Diagnosis",
            "left_path" -> "taking,pills,paracetamol",
            "right_path" -> "taking,disease")),
        Annotation(
          NODE,
          15,
          20,
          "taking",
          Map(
            "entities" -> "Diagnosis,Diagnosis",
            "left_path" -> "taking,disease,due,to,heart",
            "right_path" -> "taking,disease"))))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  "Graph Extraction with LightPipeline" should "return dependency graphs between all entities" taggedAs SlowTest in {

    val embeddings = WordEmbeddingsModel
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    val nerSmall = NerDLModel
      .pretrained()
      .setInputCols(Array("sentence", "token", "embeddings"))
      .setOutputCol("ner")

    val nerConverter = new NerConverter()
      .setInputCols("document", "token", "ner")
      .setOutputCol("entities")

    val pos = PerceptronModel
      .pretrained()
      .setInputCols("document", "token")
      .setOutputCol("pos")

    val dependencyParser = DependencyParserModel
      .pretrained()
      .setInputCols("sentence", "pos", "token")
      .setOutputCol("dependency")

    val typedDependencyParser = TypedDependencyParserModel
      .pretrained()
      .setInputCols("dependency", "pos", "token")
      .setOutputCol("dependency_type")

    val graphExtractor = new GraphExtraction()
      .setInputCols("document", "token", "ner")
      .setOutputCol("graph")
      .setRelationshipTypes(Array("lad-PER", "lad-LOC"))

    val graphFinisher = new GraphFinisher()
      .setInputCol("graph")
      .setOutputCol("graph_finished")
      .setOutputAsArray(false)

    val graphPipeline = new Pipeline().setStages(
      Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        nerSmall,
        nerConverter,
        pos,
        dependencyParser,
        typedDependencyParser,
        graphExtractor,
        graphFinisher))

    val expectedResult =
      Seq(Annotation(NODE, 0, 0, "[(lad,flat,York),(York,flat,New)],[(lad,flat,York)]", Map()))

    val graphPipelineModel = graphPipeline.fit(emptyDataSet)
    val lightPipeline = new LightPipeline(graphPipelineModel)
    val result = lightPipeline.fullAnnotate("Peter Parker is a nice lad and lives in New York")

    assert(result("graph_finished") == expectedResult)

  }

}
