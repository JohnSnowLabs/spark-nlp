package com.johnsnowlabs.ml.openvino

import com.johnsnowlabs.ml.tensorflow.io.ChunkBytes
import com.johnsnowlabs.ml.tensorflow.sign.ModelSignatureConstants._
import com.johnsnowlabs.ml.tensorflow.sign.{ModelSignatureConstants, ModelSignatureManager}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.util.{FileHelper, ZipArchiveUtil}
import org.apache.commons.io.FileUtils
import org.intel.openvino.{CompiledModel, Core, Model}
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID
import scala.collection.JavaConverters._

class OpenvinoWrapper(modelBytes: Array[Byte], weightsBytes: Array[Array[Byte]])
    extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null, null)
  }

  // Important for serialization on none-kyro serializers
  @transient private val logger = LoggerFactory.getLogger(this.getClass.toString)
  @transient private var compiledModel: CompiledModel = _

  def getCompiledModel(
      device: String = "AUTO",
      properties: Map[String, String] = Map.empty): CompiledModel =
    this.synchronized {
      if (compiledModel == null) {
        val path = Files
          .createTempDirectory(
            UUID.randomUUID().toString.takeRight(12) + OpenvinoWrapper.ModelSuffix)
          .toAbsolutePath
          .toString
        val tmpModelPath = Paths.get(path, Openvino.modelXml)
        val tmpWeightsPath = Paths.get(path, Openvino.modelBin)

        // save the binary data of the model and weights to files
        FileUtils.writeByteArrayToFile(tmpModelPath.toFile, modelBytes)
        ChunkBytes.writeByteChunksInFile(tmpWeightsPath, weightsBytes)

        compiledModel = OpenvinoWrapper.core.compile_model(
          tmpModelPath.toAbsolutePath.toString,
          device,
          properties.asJava)

        logger.debug(
          s"Compiled OpenVINO IR model on device: $device with properties: $properties")
        FileHelper.delete(path)
      }
      compiledModel
    }

  def saveToFile(file: String): Unit = {
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ov")
      .toAbsolutePath
      .toString

    FileUtils.writeByteArrayToFile(Paths.get(tmpFolder, Openvino.modelXml).toFile, modelBytes)
    ChunkBytes.writeByteChunksInFile(Paths.get(tmpFolder, Openvino.modelBin), weightsBytes)

    ZipArchiveUtil.zip(tmpFolder, file)
    FileHelper.delete(tmpFolder)
  }

}

/** Companion object */
object OpenvinoWrapper {

  private val logger: Logger = LoggerFactory.getLogger(this.getClass.toString)
  private[OpenvinoWrapper] val core: Core = this.synchronized {
    if (core == null) {
      new Core()
    } else {
      core
    }
  }

  // size of bytes store in each chunk/array
  private val BUFFER_SIZE = 1024 * 1024

  private val ModelSuffix = "_ov_model"

  /** Reads models from supported file formats and exports them into OpenVINO Intermediate
    * Representation (IR) format. The resulting framework-independent model representation
    * consists of a model graph (.xml) and weights (.bin) files.
    *
    * @param modelPath
    *   Path to the source model
    * @param targetPath
    *   Path to the converted model directory
    * @param useBundle
    *   Read from a provided model bundle
    * @param zipped
    *   Unpack the zipped model
    */
  def convertToOpenvinoFormat(
      modelPath: String,
      targetPath: String,
      detectedEngine: String,
      useBundle: Boolean,
      zipped: Boolean = true): Unit = {
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + ModelSuffix)
      .toAbsolutePath
      .toString

    val folder =
      if (zipped) {
        ZipArchiveUtil.unzip(new File(modelPath), Some(tmpFolder))
      } else {
        modelPath
      }

    logger.debug(s"Converting the $detectedEngine model to OpenVINO Intermediate format")

    val srcModelPath: String =
      detectedEngine match {
        case TensorFlow.name =>
          folder
        case ONNX.name =>
          Paths.get(folder, ONNX.modelName).toString
      }

    val model: Model = core.read_model(srcModelPath)
    val ovModelPath = Paths.get(targetPath, Openvino.modelXml).toAbsolutePath.toString
    val ovWeightsPath = Paths.get(targetPath, Openvino.modelBin).toAbsolutePath.toString
    org.intel.openvino.Openvino.serialize(model, ovModelPath, ovWeightsPath)

    FileHelper.delete(tmpFolder)
  }

  /** Reads a model saved in the OpenVINO IR format and loads the OpenVINO model wrapper.
    *
    * @param path
    *   Path to the IR model folder
    * @param zipped
    *   Unpack zipped model
    * @param device
    *   Device to load model on
    * @param properties
    *   Properties for this load operation
    * @return
    *   The OpenVINO model wrapper and the normalized tensor name map for the model
    */
  def fromOpenvinoFormat(
      path: String,
      zipped: Boolean = true,
      device: String = "AUTO",
      properties: Map[String, String] = Map.empty): (OpenvinoWrapper, Map[String, String]) = {

    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + ModelSuffix)
      .toAbsolutePath
      .toString

    val folder =
      if (zipped)
        ZipArchiveUtil.unzip(new File(path), Some(tmpFolder))
      else
        path

    val modelPath = Paths.get(folder, Openvino.modelXml)
    val weightsPath = Paths.get(folder, Openvino.modelBin)

    logger.debug(s"Reading and compiling IR model on device: $device...")
    val modelBytes = FileUtils.readFileToByteArray(modelPath.toFile)
    val weightsBytes = ChunkBytes.readFileInByteChunks(weightsPath, BUFFER_SIZE)
    val compiledModel: CompiledModel =
      core.compile_model(modelPath.toAbsolutePath.toString, device, properties.asJava)

    val openvinoWrapper = new OpenvinoWrapper(modelBytes, weightsBytes)
    openvinoWrapper.compiledModel = compiledModel

    val tensorNamesMap = (compiledModel.outputs().asScala ++ compiledModel.inputs().asScala)
      .map(_.get_any_name())
      .map(tensorName => ModelSignatureConstants.toAdoptedKeys(tensorName) -> tensorName)
      .toMap

    FileHelper.delete(tmpFolder)
    (openvinoWrapper, tensorNamesMap)
  }
}
