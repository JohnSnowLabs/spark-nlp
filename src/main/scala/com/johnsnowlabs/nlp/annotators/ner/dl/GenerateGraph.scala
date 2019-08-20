package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp.util.io.ResourceHelper

case class GraphData(graphFilePath: String, numberOfTags: Int, embeddingsDimension: Int,
                     numberOfChars: Int, lstmSize: Int=128)

class GenerateGraph(graphData: GraphData) {

  private val useContrib: Boolean = { if (ResourceHelper.getOsName == "Windows") false else true }
  private val pythonFile = "python/sparknlp/graph.py"

  def getModelName: String = {
    val namePrefix = getNamePrefix
    s"${namePrefix}_${graphData.numberOfTags}_${graphData.embeddingsDimension}_${graphData.lstmSize}_${graphData.numberOfChars}"
  }

  private def getNamePrefix: String = {
    if (useContrib) "blstm" else "blstm-noncontrib"
  }

  def create: String = {
    import sys.process._

    val pythonScript = "python " + pythonFile + getArguments
    val stderr = new StringBuilder
    val status = pythonScript ! ProcessLogger(stdout append _, stderr append _)
    if (status == 0) {
      "Graph created successfully"
    } else {
      getErrorMessage(stderr.toString())
    }
   }

  private def getArguments: String = {
    val fileName = graphData.graphFilePath + "/" + getModelName
    " " + fileName + " " + useContrib + " " + graphData.numberOfTags + " " + graphData.embeddingsDimension + " " +
      graphData.numberOfChars
  }

  private def getErrorMessage(fullErrorMessage: String): String = {
    val pattern = "Exception:[\\sa-zA-Z\\d_.-]*".r
    val errorMessage = pattern findFirstIn fullErrorMessage
    errorMessage.getOrElse("")
  }

}
