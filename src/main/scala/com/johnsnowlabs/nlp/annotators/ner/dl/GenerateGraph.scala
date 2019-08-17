package com.johnsnowlabs.nlp.annotators.ner.dl

class GenerateGraph {

  def createGraph(pythonScript: String): String = {
    import sys.process._

    val stderr = new StringBuilder
    val status = pythonScript ! ProcessLogger(stdout append _, stderr append _)
    if (status == 0) {
      "Graph created successfully"
    } else {
      getErrorMessage(stderr.toString())
    }
   }

  def getErrorMessage(fullErrorMessage: String): String = {
    val pattern = "Exception:[\\sa-zA-Z\\d_.-]*".r
    val errorMessage = pattern findFirstIn fullErrorMessage
    errorMessage.getOrElse("")
  }

}
