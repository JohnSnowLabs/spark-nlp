package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.NODE
import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations}
import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec

import scala.collection.mutable

class GraphExtractionTest extends FlatSpec with SparkSessionTest with GraphExtractionFixture {

  spark.conf.set("spark.sql.crossJoin.enabled", "true")

  "Graph Extraction" should "return dependency graphs between all entities" taggedAs FastTest in {

    val testDataSet = getUniqueEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
      .setIncludeEdges(false)
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,TIME",
        "left_path" -> "canceled,United", "right_path" -> "canceled,flights,morning")),
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "left_path" -> "canceled,United", "right_path" -> "canceled,flights,Houston")),
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "TIME,LOC",
        "left_path" -> "canceled,flights,morning", "right_path" -> "canceled,flights,Houston"))
    ))

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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "left_path" -> "canceled,United", "right_path" -> "canceled,flights,Houston"))
    ))

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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "left_path" -> "canceled,United", "right_path" -> "canceled,flights,Houston")),
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,TIME",
        "left_path" -> "canceled,United", "right_path" -> "canceled,flights,morning"))
    ))

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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "left_path" -> "canceled,United", "right_path" -> "canceled,flights,Houston")),
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "left_path" -> "canceled,United",
        "right_path" -> "canceled,flights,Houston,Dallas"))
    ))

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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "LOC,LOC",
        "left_path" -> "canceled,flights,Houston,Dallas",
        "right_path" -> "canceled,flights,Houston"))
    ))

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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "LOC,TIME",
        "left_path" -> "canceled,flights,Houston", "right_path" -> "canceled,flights,morning")),
      Annotation(NODE, 59, 60, "go", Map("entities" -> "LOC,TIME",
        "left_path" -> "go,London",
        "right_path" -> "go,tomorrow"))
    ))

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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "LOC,TIME",
        "left_path" -> "canceled,flights,Houston", "right_path" -> "canceled,flights,morning"))
    ))

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

  it should "handle overlapping entities" ignore {
    //Ignored because it downloads POS and Dependency Parser pretrained models
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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 51, 54, "goes", Map("entities" -> "PER,LOC",
        "left_path" -> "goes,Bill", "right_path" -> "goes,Pasadena"))
    ))

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

    val expectedGraph = Array(Seq(Annotation(NODE, 32, 35, "sees", Map("relationship" -> "sees,PER",
      "path1" -> "sees,nsubj,John", "path2" -> "sees,ccomp,goes,nsubj,Bill",
      "path3" -> "sees,ccomp,goes,nsubj,Bill,conj,Mary"))
    ))

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

    val result = graphDataSet.select("graph").rdd.map(row => row(0).asInstanceOf[mutable.WrappedArray[String]])
      .collect().toList
    assert(result.head.isEmpty)
  }

  it should "find paths between relationship types for several relationships" taggedAs FastTest in {
    val testDataSet = getDeepEntities(spark, tokenizerPipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("document", "token", "entities")
      .setOutputCol("graph")
      .setRelationshipTypes(Array("goes-PER", "goes-LOC"))
      .setIncludeEdges(false)
    val expectedGraph = Array(Seq(
      Annotation(NODE, 51, 54, "goes", Map("relationship" -> "goes,PER",
        "path1" -> "goes,Bill", "path2" -> "goes,Bill,Mary")),
      Annotation(NODE, 51, 54, "goes", Map("relationship" -> "goes,LOC",
        "path1" -> "goes,Pasadena"))
    ))

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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 32, 44, "polymorphisms", Map("relationship" -> "polymorphisms,GENE",
        "path1" -> "polymorphisms,nsubj,Influence,nmod,gene,amod,interleukin-6")),
      Annotation(NODE, 32, 44, "polymorphisms", Map("relationship" -> "polymorphisms,DISEASE",
        "path1" -> "polymorphisms,nmod,coronary_artery_calcification"))
    ))

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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,TIME",
        "left_path" -> "canceled,nsubj,United", "right_path" -> "canceled,obj,flights,compound,morning")),
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "left_path" -> "canceled,nsubj,United", "right_path" -> "canceled,obj,flights,nmod,Houston")),
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "TIME,LOC",
        "left_path" -> "canceled,obj,flights,compound,morning",
        "right_path" -> "canceled,obj,flights,nmod,Houston"))
    ))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

}
