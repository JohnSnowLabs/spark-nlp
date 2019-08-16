package com.johnsnowlabs.nlp.annotators.ner.dl

import org.scalatest.FlatSpec

class GenerateGraphTestSpec extends FlatSpec {

  "GenerateGraph" should "execute python script" in {
    val generateGraph = new GenerateGraph()
    val pythonScript = "/sparknlp/graph.py 80 200 125"

    generateGraph.createGraph(pythonScript)
  }

}
