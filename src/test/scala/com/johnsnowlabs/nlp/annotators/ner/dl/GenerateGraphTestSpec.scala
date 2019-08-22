package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.sql.SparkSession
import org.scalatest.FlatSpec

class GenerateGraphTestSpec extends FlatSpec {

  private val numberOfTags = 80
  private val embeddingsDimension = 200
  private val numberOfChars = 125
  private val spark = SparkSession.builder()
    .appName("benchmark")
    .master("local[*]")
    .config("spark.driver.memory", "1G")
    .config("spark.kryoserializer.buffer.max", "200M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  "GenerateGraph" should "get model file name" in {
    val graphData = GraphParams(numberOfTags, embeddingsDimension, numberOfChars)
    val generateGraph = new GenerateGraph(graphData, "", spark)
    val graphFileName = generateGraph.getModelName

    if (ResourceHelper.getOsName == "Linux") {
      assert(graphFileName == "blstm_80_200_128_125")
    } else {
      assert(graphFileName == "blstm-noncontrib_80_200_128_125")
    }
  }

  "GenerateGraph" should "create a graph" in {
    val graphFilePath = "./tmp"
    val graphParams = GraphParams(numberOfTags, embeddingsDimension, numberOfChars)
    val generateGraph = new GenerateGraph(graphParams, graphFilePath, spark)
    val expectedMessage = "Graph created successfully"

    val message = generateGraph.create

    assert(message==expectedMessage)
  }

  "GenerateGraph" should "load a graph from default path" in {
    val numberOfTags = 10
    val embeddingsDimension = 100
    val numberOfChars = 100
    val graphFilePath = "./tmp"
    val graphParams = GraphParams(numberOfTags, embeddingsDimension, numberOfChars)
    val generateGraph = new GenerateGraph(graphParams, graphFilePath, spark)

    generateGraph.loadModel()

  }

  "GenerateGraph with wrong argument" should "return an error message" in {
    val graphFilePath = "./tmp"
    val graphParams = GraphParams(-1, embeddingsDimension, numberOfChars)
    val generateGraph = new GenerateGraph(graphParams, graphFilePath, spark)
    val expectedString = "Error:"

    val message = generateGraph.create
    assert(message contains expectedString)
  }



}
