/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.openvino

import com.johnsnowlabs.ml.tensorflow.io.ChunkBytes
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader, FileHelper, ZipArchiveUtil}
import org.apache.commons.io.FileUtils
import org.intel.openvino.{CompiledModel, Core, Model}
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID
import scala.collection.JavaConverters._

class OpenvinoWrapper(
    modelBytes: Array[Byte],
    weightsBytes: Array[Array[Byte]],
    modelPath: Option[String] = None)
    extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null, null)
  }

  // Important for serialization on none-kyro serializers
  @transient private var compiledModel: CompiledModel = _

  def getCompiledModel(
      properties: Map[String, String] = Map.empty[String, String]): CompiledModel =
    this.synchronized {
      if (compiledModel == null) {
        compiledModel = OpenvinoWrapper.withSafeOvModelLoader(
          modelBytes,
          weightsBytes,
          modelPath,
          properties = properties)
      }
      compiledModel
    }

  def saveToFile(file: String): Unit = {
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ov")
      .toAbsolutePath
      .toString

    val fileName = Paths.get(file).getFileName.toString
    FileUtils.writeByteArrayToFile(Paths.get(tmpFolder, s"${fileName}.xml").toFile, modelBytes)
    ChunkBytes.writeByteChunksInFile(Paths.get(tmpFolder, s"${fileName}.bin"), weightsBytes)

    ZipArchiveUtil.zip(tmpFolder, file)
    FileHelper.delete(tmpFolder)
  }

}

/** Companion object */
object OpenvinoWrapper {

  private val logger: Logger = LoggerFactory.getLogger(this.getClass.toString)
  private[OpenvinoWrapper] val core: Core = new Core

  // size of bytes store in each chunk/array
  private val BUFFER_SIZE = 1024 * 1024

  private val ModelSuffix = "_ov_model"

  /** Read the model from the given path, unpack if zipped, and return the loaded OpenvinoWrapper.
    * If source model is not in OpenVINO format, it is converted first.
    *
    * @param path
    *   Path to the model
    * @param modelName
    *   The model filename
    * @param zipped
    *   Unpack zipped model
    * @param detectedEngine
    *   The source model format
    * @param properties
    *   Properties for this load operation
    * @return
    *   The resulting OpenVINO model wrapper
    */
  def read(
      path: String,
      modelName: String = Openvino.ovModel,
      zipped: Boolean = true,
      detectedEngine: String = Openvino.name,
      properties: Map[String, String] = Map.empty): OpenvinoWrapper = {

    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + ModelSuffix)
      .toAbsolutePath
      .toString

    val folder =
      if (zipped)
        ZipArchiveUtil.unzip(new File(path), Some(tmpFolder))
      else
        path

    val (modelPath, weightsPath) =
      detectedEngine match {
        case TensorFlow.name =>
          val model: Model = core.read_model(folder)
          val ovModelPath = Paths.get(tmpFolder, s"${Openvino.ovModel}.xml")
          val ovWeightsPath = Paths.get(tmpFolder, s"${Openvino.ovModel}.bin")
          org.intel.openvino.Openvino
            .save_model(model, ovModelPath.toAbsolutePath.toString, false)
          (ovModelPath, ovWeightsPath)
        case ONNX.name =>
          val model: Model = core.read_model(Paths.get(folder, ONNX.modelName).toString)
          val ovModelPath = Paths.get(tmpFolder, s"${Openvino.ovModel}.xml")
          val ovWeightsPath = Paths.get(tmpFolder, s"${Openvino.ovModel}.bin")
          org.intel.openvino.Openvino
            .save_model(model, ovModelPath.toAbsolutePath.toString, false)
          (ovModelPath, ovWeightsPath)
        case _ =>
          (Paths.get(folder, s"$modelName.xml"), Paths.get(folder, s"$modelName.bin"))
      }

    val modelBytes = FileUtils.readFileToByteArray(modelPath.toFile)
    val weightsBytes = ChunkBytes.readFileInByteChunks(weightsPath, BUFFER_SIZE)
    val device = ConfigLoader.getConfigStringValue(ConfigHelper.openvinoDevice)
    val compiledModel: CompiledModel = withSafeOvModelLoader(
      modelBytes,
      weightsBytes,
      Some(modelPath.toAbsolutePath.toString),
      device,
      properties)

    val openvinoWrapper = new OpenvinoWrapper(modelBytes, weightsBytes)
    openvinoWrapper.compiledModel = compiledModel

    FileHelper.delete(tmpFolder)
    openvinoWrapper
  }

  /** Prepare the model for inference by compiling into a device-specific graph representation.
    * Returns the compiled model object.
    *
    * @param modelBytes
    *   Model xml file as byte array
    * @param weightsBytes
    *   The model weights as byte array
    * @param modelPath
    *   Optional path to the model directory
    * @param device
    *   Device to compile the model to
    * @param properties
    *   Properties for this load operation
    * @return
    *   Object representing the compiled model
    */
  def withSafeOvModelLoader(
      modelBytes: Array[Byte],
      weightsBytes: Array[Array[Byte]],
      modelPath: Option[String] = None,
      device: String = "CPU",
      properties: Map[String, String]): CompiledModel = {
    logger.info(s"Compiling OpenVINO model to device: $device")
    if (modelPath.isDefined) {
      val compiledModel = core.compile_model(modelPath.get, device, properties.asJava)
      compiledModel
    } else {
      val path = Files
        .createTempDirectory(
          UUID.randomUUID().toString.takeRight(12) + OpenvinoWrapper.ModelSuffix)
        .toAbsolutePath
        .toString
      val tmpModelPath = Paths.get(path, s"${Openvino.ovModel}.xml")
      val tmpWeightsPath = Paths.get(path, s"${Openvino.ovModel}.bin")

      // save the binary data of the model and weights to files
      FileUtils.writeByteArrayToFile(tmpModelPath.toFile, modelBytes)
      ChunkBytes.writeByteChunksInFile(tmpWeightsPath, weightsBytes)

      val xmlPath = tmpModelPath.toAbsolutePath.toString
      val compiledModel = core.compile_model(xmlPath, device, properties.asJava)

      FileHelper.delete(path)
      compiledModel
    }
  }

  case class EncoderDecoderWrappers(
      encoder: OpenvinoWrapper,
      decoder: OpenvinoWrapper,
      decoderWithPast: OpenvinoWrapper)
}
