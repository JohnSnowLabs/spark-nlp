package com.johnsnowlabs.nlp.annotators.ner.dl

import org.scalatest.FlatSpec

class GenerateGraphTestSpec extends FlatSpec {

  "GenerateGraph" should "create a graph" in {
    val generateGraph = new GenerateGraph()
    val pythonScript = "python ./graph.py 80 200 125"
    val expectedMessage = "Graph created successfully"

    val message = generateGraph.createGraph(pythonScript)

    assert(message==expectedMessage)
  }

  "GenerateGraph with wrong argument" should "return an error message" in {
    val generateGraph = new GenerateGraph()
    val pythonScript = "python ./graph.py -1 200 125"
    val expectedMessage = "Exception: Error message"

    val message = generateGraph.createGraph(pythonScript)
    assert(message==expectedMessage)
  }

}
