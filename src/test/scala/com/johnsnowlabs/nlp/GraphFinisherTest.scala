package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.{GraphExtraction, GraphExtractionFixture, SparkSessionTest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Dataset
import org.scalatest.FlatSpec

import scala.collection.mutable

class GraphFinisherTest extends FlatSpec with SparkSessionTest with GraphExtractionFixture {

  import spark.implicits._

  private val textDataSet = Seq("Bruce Wayne lives in Gotham").toDS.toDF("text")

  "GraphFinisher" should "raise an error when NODE annotator type column does not exist" in {

    val finisher = new GraphFinisher().setInputCol("token").setOutputCol("finisher")
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, finisher))

    assertThrows[IllegalArgumentException] {
      pipeline.fit(textDataSet)
    }

  }

  it should "output graph in RDF format as array" in {
    val testDataSet = getDeepEntities(spark, tokenizerWithSentencePipeline)
    testDataSet.printSchema()
    val graphExtractor = new GraphExtraction()
      .setInputCols("document", "token", "entities")
      .setOutputCol("graph")
      .setRelationshipTypes(Array("sees-PER"))
    val finisher = new GraphFinisher().setInputCol("graph").setOutputCol("finisher")
    val pipeline = new Pipeline().setStages(Array(graphExtractor, finisher))
    val expectedGraph = List(
      Seq(Seq("sees", "nsubj", "John")),
      Seq(Seq("sees", "ccomp", "goes"), Seq("goes", "nsubj", "Bill")),
      Seq(Seq("sees", "ccomp", "goes"), Seq("goes", "nsubj", "Bill"), Seq("Bill", "conj", "Mary"))
    )

    val graphDataSet = pipeline.fit(testDataSet).transform(testDataSet)
    graphDataSet.show(false)
    val actualGraph = getFinisherAsArray(graphDataSet)

    assert(actualGraph == expectedGraph)
  }

  it should "output graph in RDF format as string" in {
    val testDataSet = getDeepEntities(spark, tokenizerWithSentencePipeline)
    val graphExtractor = new GraphExtraction()
      .setInputCols("document", "token", "entities")
      .setOutputCol("graph")
      .setRelationshipTypes(Array("sees-PER"))
    val finisher = new GraphFinisher().setInputCol("graph").setOutputCol("finisher").setOutputAsArray(false)
    val pipeline = new Pipeline().setStages(Array(graphExtractor, finisher))
    val expectedGraph = List(Seq(Seq("(sees,nsubj,John)"),
      Seq("(sees,ccomp,goes)", "(goes,nsubj,Bill)"),
      Seq("(sees,ccomp,goes)", "(goes,nsubj,Bill)", "(Bill,conj,Mary)")
    ))

    val graphDataSet = pipeline.fit(testDataSet).transform(testDataSet)
    val actualGraph = getFinisher(graphDataSet)

    assert(actualGraph == expectedGraph)
  }

  private def getFinisherAsArray(dataSet: Dataset[_]) = {
    val paths = dataSet.select("finisher").rdd.map{row =>
      val result: Seq[Seq[String]] = row.get(0).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[String]]]
      result
    }.collect().toList
    paths.flatten
  }

  private def getFinisher(dataSet: Dataset[_]) = {
    val paths = dataSet.select("finisher").rdd.map{row =>
      val result: Seq[String] = row.get(0).asInstanceOf[mutable.WrappedArray[String]]
      result
    }.collect().toList
    paths
  }

}
