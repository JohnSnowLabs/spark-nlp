package com.johnsnowlabs.nlp.annotators.ner.dl

import org.scalatest.FlatSpec

class GenerateGraphTestSpec extends FlatSpec {

  "GenerateGraph" should "create a graph" in {
    val generateGraph = new GenerateGraph()
    val pythonFile = "python/sparknlp/graph.py"
    val graphFilePath = "src/main/resources/ner-dl"
    val numberOfTags = 80
    val embeddings_dimension = 200
    val numberOfChars = 125
    val arguments = " " + graphFilePath + " " + numberOfTags + " " + embeddings_dimension + " " + numberOfChars
    val pythonScript = "python " + pythonFile + arguments
    val expectedMessage = "Graph created successfully"

    val message = generateGraph.create(pythonScript)

    assert(message==expectedMessage)
  }

  "GenerateGraph with wrong argument" should "return an error message" ignore {
    val generateGraph = new GenerateGraph()
    val pythonScript = "python ./graph.py -1 200 125"
    val expectedMessage = "Exception: Error message"

    val message = generateGraph.create(pythonScript)
    assert(message==expectedMessage)
  }

}
