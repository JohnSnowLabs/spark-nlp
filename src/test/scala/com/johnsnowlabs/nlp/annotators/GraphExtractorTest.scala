package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.VERTEX
import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations}
import org.scalatest.FlatSpec

class GraphExtractorTest extends FlatSpec with SparkSessionTest with GraphExtractorFixture {

  "Graph Extractor" should "return dependency graphs between all entities" in {

    val testDataSet = getUniqueEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtractor()
      .setInputCols("sentence", "token", "heads", "deprel", "entities")
      .setOutputCol("graph")
    val expectedGraph = Array(Seq(
      Annotation(VERTEX, 7, 14, "canceled", Map("entities" -> "ORG,TIME",
        "path" -> "canceled,United,canceled,flights,morning",
        "left_path" -> "canceled->United", "right_path" -> "canceled->flights->morning")),
      Annotation(VERTEX, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "path" -> "canceled,United,canceled,flights,Houston",
        "left_path" -> "canceled->United", "right_path" -> "canceled->flights->Houston")),
      Annotation(VERTEX, 7, 14, "canceled", Map("entities" -> "TIME,LOC",
        "path" -> "canceled,flights,morning,canceled,flights,Houston",
        "left_path" -> "canceled->flights->morning", "right_path" -> "canceled->flights->Houston"))
    ))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)

  }

  it should "return dependency graphs for a pair of entities" in {
    val testDataSet = getUniqueEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtractor()
      .setInputCols("sentence", "token", "heads", "deprel", "entities")
      .setOutputCol("graph")
      .setEntityRelationships(Array("ORG-LOC"))
    val expectedGraph = Array(Seq(
      Annotation(VERTEX, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "path" -> "canceled,United,canceled,flights,Houston",
        "left_path" -> "canceled->United", "right_path" -> "canceled->flights->Houston")),
    ))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "return dependency graphs for a subset of entities" in {
    val testDataSet = getUniqueEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtractor()
      .setInputCols("sentence", "token", "heads", "deprel", "entities")
      .setOutputCol("graph")
      .setEntityRelationships(Array("ORG-LOC", "ORG-TIME"))
    val expectedGraph = Array(Seq(
      Annotation(VERTEX, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "path" -> "canceled,United,canceled,flights,Houston",
        "left_path" -> "canceled->United", "right_path" -> "canceled->flights->Houston")),
      Annotation(VERTEX, 7, 14, "canceled", Map("entities" -> "ORG,TIME",
        "path" -> "canceled,United,canceled,flights,morning",
        "left_path" -> "canceled->United", "right_path" -> "canceled->flights->morning"))
    ))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "return dependency graphs when entities are ambiguous" in {
    val testDataSet = getAmbiguousEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtractor()
      .setInputCols("sentence", "token", "heads", "deprel", "entities")
      .setOutputCol("graph")
      .setEntityRelationships(Array("ORG-LOC"))
    val expectedGraph = Array(Seq(
      Annotation(VERTEX, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "path" -> "canceled,United,canceled,flights,Houston",
        "left_path" -> "canceled->United", "right_path" -> "canceled->flights->Houston")),
      Annotation(VERTEX, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "path" -> "canceled,United,canceled,flights,Houston,Dallas",
        "left_path" -> "canceled->United",
        "right_path" -> "canceled->flights->Houston->Dallas"))
    ))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)

  }

  it should "exclude 0 length paths" in {
    val testDataSet = getAmbiguousEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtractor()
      .setInputCols("sentence", "token", "heads", "deprel", "entities")
      .setOutputCol("graph")
      .setEntityRelationships(Array("LOC-LOC"))
    val expectedGraph = Array(Seq(
      Annotation(VERTEX, 7, 14, "canceled", Map("entities" -> "LOC,LOC",
        "path" -> "canceled,flights,Houston,Dallas,canceled,flights,Houston",
        "left_path" -> "canceled->flights->Houston->Dallas",
        "right_path" -> "canceled->flights->Houston"))
    ))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "extract graphs for each sentence" in {
    val testDataSet = getEntitiesFromTwoSentences(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtractor()
      .setInputCols("sentence", "token", "heads", "deprel", "entities")
      .setOutputCol("graph")
      .setEntityRelationships(Array("LOC-TIME"))
    val expectedGraph = Array(Seq(
      Annotation(VERTEX, 7, 14, "canceled", Map("entities" -> "LOC,TIME",
        "path" -> "canceled,flights,Houston,canceled,flights,morning",
        "left_path" -> "canceled->flights->Houston", "right_path" -> "canceled->flights->morning")),
      Annotation(VERTEX, 59, 60, "go", Map("entities" -> "LOC,TIME",
        "path" -> "go,London,go,tomorrow",
        "left_path" -> "go->London",
        "right_path" -> "go->tomorrow"))
    ))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "filter a sentence when filtering whit min sentence parameter" in {

    val testDataSet = getEntitiesFromTwoSentences(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtractor()
      .setInputCols("sentence", "token", "heads", "deprel", "entities")
      .setOutputCol("graph")
      .setEntityRelationships(Array("LOC-TIME"))
      .setMinSentenceSize(33)
    val expectedGraph = Array(Seq(
      Annotation(VERTEX, 7, 14, "canceled", Map("entities" -> "LOC,TIME",
        "path" -> "canceled,flights,Houston,canceled,flights,morning",
        "left_path" -> "canceled->flights->Houston", "right_path" -> "canceled->flights->morning"))
    ))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "return empty VERTEX token when filtering all long sentence whit max sentence parameter" in {
    val testDataSet = getUniqueEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtractor()
      .setInputCols("sentence", "token", "heads", "deprel", "entities")
      .setOutputCol("graph")
      .setEntityRelationships(Array("ORG-LOC"))
      .setMaxSentenceSize(5)
    val expectedGraph = Array(Seq(Annotation(VERTEX, 0, 0, "", Map())))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

}
