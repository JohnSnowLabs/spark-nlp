package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.scalatest.FlatSpec

class GenerateGraphTestSpec extends FlatSpec {

  private val numberOfTags = 80
  private val embeddingsDimension = 200
  private val numberOfChars = 125

  "GenerateGraph" should "get model file name" in {
    val graphData = GraphData("", numberOfTags, embeddingsDimension, numberOfChars)
    val generateGraph = new GenerateGraph(graphData)
    val graphFileName = generateGraph.getModelName

    if (ResourceHelper.getOsName == "Linux") {
      assert(graphFileName == "blstm_80_200_128_125")
    } else {
      assert(graphFileName == "blstm-noncontrib_80_200_128_125")
    }
  }

  "GenerateGraph" should "create a graph" in {
    val graphFilePath = "src/main/resources/ner-dl"
    val graphData = GraphData(graphFilePath, numberOfTags, embeddingsDimension, numberOfChars)
    val generateGraph = new GenerateGraph(graphData)
    val expectedMessage = "Graph created successfully"

    val message = generateGraph.create

    assert(message==expectedMessage)
  }

  "GenerateGraph with wrong argument" should "return an error message" ignore {
    //val generateGraph = new GenerateGraph()
    val pythonScript = "python ./graph.py -1 200 125"
    val expectedMessage = "Exception: Error message"

//    val message = generateGraph.create(pythonScript)
//    assert(message==expectedMessage)
  }



}
