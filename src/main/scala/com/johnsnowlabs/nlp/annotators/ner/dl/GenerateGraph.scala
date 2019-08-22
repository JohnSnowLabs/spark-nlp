package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

case class GraphParams(numberOfTags: Int, embeddingsDimension: Int, numberOfChars: Int, lstmSize: Int=128)

class GenerateGraph(graphParams: GraphParams, graphFilePath: String, sparkSession: SparkSession) {

  private val useContrib: Boolean = { if (ResourceHelper.getOsName == "Windows") false else true }
  private val pythonFile = "python/sparknlp/graph.py"
  private val defaultPath = "src/main/resources/ner-dl"

  def getModelName: String = {
    val namePrefix = getNamePrefix
    s"${namePrefix}_${graphParams.numberOfTags}_${graphParams.embeddingsDimension}_${graphParams.lstmSize}_${graphParams.numberOfChars}"
  }

  private def getNamePrefix: String = {
    if (useContrib) "blstm" else "blstm-noncontrib"
  }

  def loadModel(): Unit = {
    val modelName = getModelName
    if (fileExists(graphFilePath + "/" + modelName + ".pb")) {
      println("Load Model")
    } else {
      if (fileExists(defaultPath + "/" + modelName + ".pb")) {
        println("Load model from default Path")
      } else {
        create
        println("Load model after creation")
      }
    }
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
    val fileName = graphFilePath + "/" + getModelName
    " " + fileName + " " + useContrib + " " + graphParams.numberOfTags + " " + graphParams.embeddingsDimension +
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
