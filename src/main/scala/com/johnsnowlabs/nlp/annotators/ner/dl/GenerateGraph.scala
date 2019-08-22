package com.johnsnowlabs.nlp.annotators.ner.dl

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

case class GraphParams(numberOfTags: Int, embeddingsDimension: Int, numberOfChars: Int, lstmSize: Int=128)

class GenerateGraph(sparkSession: SparkSession, graphParams: GraphParams, useContrib: Boolean, graphFilePath: String, pythonLauncher: String, pythonGraphFile: String) {

  def getModelName: String = {
    val namePrefix = getNamePrefix
    s"${namePrefix}_${graphParams.numberOfTags}_${graphParams.embeddingsDimension}_${graphParams.lstmSize}_${graphParams.numberOfChars}"
  }

  private def getNamePrefix: String = {
    if (useContrib) "blstm" else "blstm-noncontrib"
  }

  def createModel(): Unit = {
    val modelName = getModelName
    if (!fileExists(graphFilePath + "/" + modelName + ".pb")) {
      val message = create
      if ((message contains "Error:") || (message contains "Exception:")) {
        throw new Exception(s"Not able to create model ${modelName} in path ${graphFilePath}." +
          s"\nEmbedded Error Message:\n$message")
      }
    }
  }

  def create: String = {
    import sys.process._

    val pythonScript = pythonLauncher + " " + pythonGraphFile + getArguments
    val stderr = new StringBuilder
    val status = pythonScript ! ProcessLogger(stdout append "\n"+_, stderr append "\n"+_)
    if (status == 0) {
      "Graph successfully created from JVM"
    } else {
      getErrorMessage(stderr.toString())
    }
   }

  private def getArguments: String = {
    " " + graphFilePath + " " + getModelName + " " + {if (useContrib) "1" else "0"} + " " + graphParams.numberOfTags + " " + graphParams.embeddingsDimension +
      " " + graphParams.numberOfChars
  }

  private def getErrorMessage(fullErrorMessage: String): String = {
    var pattern = "Exception:[\\sa-zA-Z\\d_.-]*".r
    var errorMessage = pattern findFirstIn fullErrorMessage
    if (errorMessage.isEmpty ) {
      pattern = "Error:[\\sa-zA-Z\\d=>_.-]*".r
      errorMessage = pattern findFirstIn fullErrorMessage
    }
    errorMessage.getOrElse("")
  }

  private def fileExists(fullPath: String): Boolean = {
    val uri = new java.net.URI(fullPath.replaceAllLiterally("\\", "/"))
    val fs: FileSystem = FileSystem.get(uri, sparkSession.sparkContext.hadoopConfiguration)
    val dataPath = new Path(fullPath)
    fs.exists(dataPath)
  }


}
