package com.johnsnowlabs.ml.pytorch

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.commons.io.FileUtils

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID

class PytorchWrapper(val modelBytes: Array[Byte]) extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null) //TODO: Check if this is really required
  }

  def saveToFile(file: String): String = {
    // 1. Create tmp director
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ner")
      .toAbsolutePath.toString

    //2. Save torchscript
    val torchScriptFile = Paths.get(tmpFolder, file).toString
    FileUtils.writeByteArrayToFile(new File(torchScriptFile), modelBytes)

    tmpFolder
  }

}

object PytorchWrapper {

  def apply(pyTorchModelPath: String, readingStrategy: String = ""): PytorchWrapper = {
    val modelBytes = readingStrategy match {
      case "local" => readBytesFromLocalFile(pyTorchModelPath)
      case _ => readBytes(pyTorchModelPath)
    }
    new PytorchWrapper(modelBytes)
  }

  private def readBytes(pyTorchModelPath: String): Array[Byte] = {
    val modelFile = new File(pyTorchModelPath).list().filter(file => file.contains(".pt")).head
    val sourceStream = ResourceHelper.SourceStream(pyTorchModelPath + modelFile)

    val inputStreamModel = sourceStream.pipe.head
    val modelBytes = new Array[Byte](inputStreamModel.available())
    inputStreamModel.read(modelBytes)

    modelBytes
  }

  private def readBytesFromLocalFile(pyTorchModelPath: String): Array[Byte] = {
    val pytorchModelBytes = FileUtils.readFileToByteArray(new File(pyTorchModelPath))
    pytorchModelBytes
  }
}
