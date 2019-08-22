package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.commons.lang.SystemUtils
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
    val graphParams = GraphParams(numberOfTags, embeddingsDimension, numberOfChars)
    val generateGraph = new GenerateGraph(spark, graphParams, false, "",
      pythonLauncher="python",
      pythonGraphFile="python/tensorflow/ner/ner_graph.py")
    val graphFileName = generateGraph.getModelName

    if (SystemUtils.IS_OS_WINDOWS) {
      assert(graphFileName == "blstm-noncontrib_80_200_128_125")
    } else {
      assert(graphFileName == "blstm_80_200_128_125")
    }
  }

  "GenerateGraph" should "create a graph" in {
    val graphFilePath = "./tmp"
    val graphParams = GraphParams(numberOfTags, embeddingsDimension, numberOfChars)
    val generateGraph = new GenerateGraph(spark, graphParams, false, graphFilePath,
      pythonLauncher="python",
      pythonGraphFile="python/tensorflow/ner/ner_graph.py")
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
    val generateGraph = new GenerateGraph(spark, graphParams, false, graphFilePath,
      pythonLauncher="python",
      pythonGraphFile="python/tensorflow/ner/ner_graph.py")

    generateGraph.createModel()

  }

  "GenerateGraph with wrong argument" should "return an error message" in {
    val graphFilePath = "./tmp"
    val graphParams = GraphParams(-1, embeddingsDimension, numberOfChars)
    val generateGraph = new GenerateGraph(spark, graphParams, false, graphFilePath,
      pythonLauncher="python",
      pythonGraphFile="python/tensorflow/ner/ner_graph.py")
    val expectedString = "Error:"

    val message = generateGraph.create
    assert(message contains expectedString)
  }



}
