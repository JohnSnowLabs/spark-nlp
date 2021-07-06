package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.NODE
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec

class GraphExtractionTest extends FlatSpec with SparkSessionTest with GraphExtractionFixture {

  "Graph Extraction" should "return dependency graphs between all entities" taggedAs FastTest in {

    val testDataSet = getUniqueEntitiesDataSet(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setExplodeEntities(true)
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,TIME",
        "path" -> "canceled,United,canceled,flights,morning",
        "left_path" -> "canceled->United", "right_path" -> "canceled->flights->morning")),
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "path" -> "canceled,United,canceled,flights,Houston",
        "left_path" -> "canceled->United", "right_path" -> "canceled->flights->Houston")),
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "TIME,LOC",
        "path" -> "canceled,flights,morning,canceled,flights,Houston",
        "left_path" -> "canceled->flights->morning", "right_path" -> "canceled->flights->Houston"))
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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "path" -> "canceled,United,canceled,flights,Houston",
        "left_path" -> "canceled->United", "right_path" -> "canceled->flights->Houston")),
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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "path" -> "canceled,United,canceled,flights,Houston",
        "left_path" -> "canceled->United", "right_path" -> "canceled->flights->Houston")),
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,TIME",
        "path" -> "canceled,United,canceled,flights,morning",
        "left_path" -> "canceled->United", "right_path" -> "canceled->flights->morning"))
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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "path" -> "canceled,United,canceled,flights,Houston",
        "left_path" -> "canceled->United", "right_path" -> "canceled->flights->Houston")),
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "ORG,LOC",
        "path" -> "canceled,United,canceled,flights,Houston,Dallas",
        "left_path" -> "canceled->United",
        "right_path" -> "canceled->flights->Houston->Dallas"))
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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "LOC,LOC",
        "path" -> "canceled,flights,Houston,Dallas,canceled,flights,Houston",
        "left_path" -> "canceled->flights->Houston->Dallas",
        "right_path" -> "canceled->flights->Houston"))
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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "LOC,TIME",
        "path" -> "canceled,flights,Houston,canceled,flights,morning",
        "left_path" -> "canceled->flights->Houston", "right_path" -> "canceled->flights->morning")),
      Annotation(NODE, 59, 60, "go", Map("entities" -> "LOC,TIME",
        "path" -> "go,London,go,tomorrow",
        "left_path" -> "go->London",
        "right_path" -> "go->tomorrow"))
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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 7, 14, "canceled", Map("entities" -> "LOC,TIME",
        "path" -> "canceled,flights,Houston,canceled,flights,morning",
        "left_path" -> "canceled->flights->Houston", "right_path" -> "canceled->flights->morning"))
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
    val expectedGraph = Array(Seq(Annotation(NODE, 0, 0, "", Map())))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "handle overlapping entities" ignore {
    val testDataSet = getOverlappingEntities(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("sentence", "token", "entities")
      .setOutputCol("graph")
      .setMergeEntities(true)

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
    val expectedGraph = Array(Seq(
      Annotation(NODE, 51, 54, "goes", Map("entities" -> "PER,LOC",
        "path" -> "goes,Bill,goes,Pasadena",
        "left_path" -> "goes->Bill", "right_path" -> "goes->Pasadena"))
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

    val expectedGraph = Array(Seq(
      Annotation(NODE, 32, 35, "sees", Map("relationship" -> "sees,PER",
        "path" -> "sees,John,sees,goes,Bill,sees,goes,Bill,Mary"))
    ))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

  it should "find paths between relationship types for several relationships" taggedAs FastTest in {
    val testDataSet = getDeepEntities(spark, tokenizerPipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("document", "token", "entities")
      .setOutputCol("graph")
      .setRelationshipTypes(Array("goes-PER", "goes-LOC"))

    val expectedGraph = Array(Seq(
      Annotation(NODE, 51, 54, "goes", Map("relationship" -> "goes,PER",
        "path" -> "goes,Bill,goes,Bill,Mary")),
      Annotation(NODE, 51, 54, "goes", Map("relationship" -> "goes,LOC",
        "path" -> "goes,Pasadena"))
    ))

    val graphDataSet = graphExtractor.transform(testDataSet)

    val actualGraph = AssertAnnotations.getActualResult(graphDataSet, "graph")
    AssertAnnotations.assertFields(expectedGraph, actualGraph)
  }

}
