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

import com.johnsnowlabs.ml.util.LoadExternalModel.notSupportedEngineError
import com.johnsnowlabs.ml.util.{LoadExternalModel, ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.util.{FileHelper, ZipArchiveUtil}
import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.apache.spark.SparkFiles
import org.apache.spark.sql.SparkSession
import org.intel.openvino.Openvino.save_model
import org.intel.openvino.{CompiledModel, Core, Model}
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import java.nio.file.{Files, Path, Paths}
import java.util.UUID
import scala.collection.JavaConverters._

class OpenvinoWrapper(var modelName: Option[String] = None) extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null)
  }

  // Important for serialization on none-kyro serializers
  @transient private var compiledModel: CompiledModel = _

  def deleteCompiledModel(): Unit = {
    this.synchronized {
      if (compiledModel != null) {
        compiledModel = null
      }
    }
  }

  def getCompiledModel(
      properties: Map[String, String] = Map.empty[String, String]): CompiledModel =
    this.synchronized {
      if (compiledModel == null) {
        val modelPath = SparkFiles.get(s"${modelName.get}.xml")
        compiledModel =
          OpenvinoWrapper.withSafeOvModelLoader(Some(modelPath), properties = properties)
      }
      compiledModel
    }

  def saveToFile(file: String): Unit = {
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ov")
      .toAbsolutePath
      .toString

    val xmlFile: String = s"${modelName.get}.xml"
    val binFile: String = s"${modelName.get}.bin"

    FileUtils.copyFile(new File(SparkFiles.get(xmlFile)), Paths.get(tmpFolder, xmlFile).toFile)
    FileUtils.copyFile(new File(SparkFiles.get(binFile)), Paths.get(tmpFolder, binFile).toFile)

    ZipArchiveUtil.zip(tmpFolder, file)
    FileHelper.delete(tmpFolder)
  }

}

/** Companion object */
object OpenvinoWrapper {

  private val logger: Logger = LoggerFactory.getLogger(this.getClass.toString)
  private[OpenvinoWrapper] val core: Core =
    try {
      new Core
    } catch {
      case e: UnsatisfiedLinkError =>
        logger.error(
          "Could not initialize OpenVINO Core. Please make sure the jsl-openvino JAR is loaded and Intel oneTBB is installed.\n" +
            "(See https://www.intel.com/content/www/us/en/docs/onetbb/get-started-guide/2021-12/overview.html)")
        throw e
    }

  private val ModelSuffix = "_ov_model"

  /** Read the model from the given path, unpack if zipped, and return the loaded OpenvinoWrapper.
    * If source model is not in OpenVINO format, it is converted first.
    *
    * @param sparkSession
    *   The Spark Session
    * @param modelPath
    *   Path to the model
    * @param modelName
    *   The model filename
    * @param zipped
    *   Unpack zipped model
    * @param useBundle
    *   Load exported model
    * @param detectedEngine
    *   The source model format
    * @param properties
    *   Properties for this load operation
    * @return
    *   The resulting OpenVINO model wrapper
    */
  def read(
      sparkSession: SparkSession,
      modelPath: String,
      modelName: String = Openvino.ovModel,
      zipped: Boolean = true,
      useBundle: Boolean = false,
      detectedEngine: String = Openvino.name,
      properties: Map[String, String] = Map.empty,
      ovFileSuffix: Option[String] = None): OpenvinoWrapper = {

    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + ModelSuffix)
      .toAbsolutePath
      .toString

    val randomSuffix = generateRandomSuffix(ovFileSuffix)
    val folder =
      if (zipped)
        ZipArchiveUtil.unzip(new File(modelPath), Some(tmpFolder), randomSuffix)
      else
        modelPath

    val (ovModelPath, ovWeightsPath) =
      detectedEngine match {
        case TensorFlow.name =>
          convertToOpenvinoFormat(folder, tmpFolder)
        case ONNX.name =>
          if (useBundle)
            convertToOpenvinoFormat(Paths.get(folder, ONNX.modelName).toString, tmpFolder)
          else
            convertToOpenvinoFormat(Paths.get(folder, s"$modelName.onnx").toString, tmpFolder)
        case Openvino.name =>
          if (useBundle)
            (Paths.get(folder, s"$modelName.xml"), Paths.get(folder, s"$modelName.bin"))
          else {
            val ovModelName = FilenameUtils.getBaseName(new File(folder).list().head)
            (Paths.get(folder, s"${ovModelName}.xml"), Paths.get(folder, s"${ovModelName}.bin"))
          }
        case _ =>
          throw new Exception(notSupportedEngineError)
      }
    sparkSession.sparkContext.addFile(ovModelPath.toString)
    sparkSession.sparkContext.addFile(ovWeightsPath.toString)

    val ovFileName = Some(FilenameUtils.getBaseName(ovModelPath.toFile.getName))
    val openvinoWrapper = new OpenvinoWrapper(ovFileName)

    val compiledModel: CompiledModel =
      withSafeOvModelLoader(Some(ovModelPath.toString), properties = properties)
    openvinoWrapper.compiledModel = compiledModel

    openvinoWrapper
  }

  private def generateRandomSuffix(fileSuffix: Option[String]): Option[String] = {
    val randomSuffix = Some(LoadExternalModel.generateRandomString(10))
    Some(s"${randomSuffix.get}${fileSuffix.getOrElse("")}")
  }

  /** Convert the model at srcPath to OpenVINO IR Format and export to exportPath.
    *
    * @param srcPath
    *   Path to the source model
    * @param exportPath
    *   Path to export converted model to
    * @param compressToFp16
    *   Whether to perform weight compression to FP16
    * @return
    *   Paths to the exported XML and BIN files
    */
  def convertToOpenvinoFormat(
      srcPath: String,
      exportPath: String,
      compressToFp16: Boolean = false): (Path, Path) = {
    logger.debug(s"Converting model from ${srcPath}, compresToFp16 = ${compressToFp16}")
    val model: Model = core.read_model(srcPath)
    val ovXmlPath = Paths.get(exportPath, s"${Openvino.ovModel}.xml")
    val ovBinPath = Paths.get(exportPath, s"${Openvino.ovModel}.bin")

    save_model(model, ovXmlPath.toAbsolutePath.toString, compressToFp16)
    (ovXmlPath, ovBinPath)
  }

  /** Prepare the model for inference by compiling into a device-specific graph representation.
    * Returns the compiled model object.
    *
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
      modelPath: Option[String] = None,
      device: String = "CPU",
      properties: Map[String, String]): CompiledModel = {
    // TODO: Let user pick inference device through Spark Config
    logger.info(s"Compiling OpenVINO model to device: $device")
    val compiledModel = core.compile_model(modelPath.get, device, properties.asJava)
    compiledModel
  }

  case class EncoderDecoderWrappers(
      encoder: OpenvinoWrapper,
      decoder: OpenvinoWrapper,
      decoderWithPast: OpenvinoWrapper)
  case class DecoderWrappers(decoder: OpenvinoWrapper)
  case class EncoderDecoderWithoutPastWrappers(encoder: OpenvinoWrapper, decoder: OpenvinoWrapper)
  case class JanusWrappers(
      languageModel: OpenvinoWrapper,
      lmHeadModel: OpenvinoWrapper,
      visionEmbeddingsModel: OpenvinoWrapper,
      textEmbeddingsModel: OpenvinoWrapper,
      mergeModel: OpenvinoWrapper,
      genHeadModel: OpenvinoWrapper,
      genEmbeddingsModel: OpenvinoWrapper,
      genDecoderModel: OpenvinoWrapper)
  case class MLLamaWrappers(
      visionEmbeddingsModel: OpenvinoWrapper,
      languageModel: OpenvinoWrapper,
      reshapeModel: OpenvinoWrapper)
  case class Qwen2VLWrappers(
      languageModel: OpenvinoWrapper,
      imageEmbedding: OpenvinoWrapper,
      imageEmbeddingMerger: OpenvinoWrapper,
      textEmbedding: OpenvinoWrapper,
      rotaryEmbedding: OpenvinoWrapper,
      patchReshapeModel: OpenvinoWrapper,
      multimodalMergeModel: OpenvinoWrapper)
  case class LLAVAWrappers(
      languageModel: OpenvinoWrapper,
      visionEmbeddingsModel: OpenvinoWrapper,
      textEmbeddingsModel: OpenvinoWrapper,
      mergeModel: OpenvinoWrapper)
  case class Phi3VWrappers(
      wte: OpenvinoWrapper,
      reshape: OpenvinoWrapper,
      languageModel: OpenvinoWrapper)
  case class SmolVLMWrappers(
      languageModel: OpenvinoWrapper,
      imageEmbedModel: OpenvinoWrapper,
      imageEncoderModel: OpenvinoWrapper,
      imageConnectorModel: OpenvinoWrapper,
      modelMergerModel: OpenvinoWrapper,
      textEmbeddingsModel: OpenvinoWrapper,
      lmHeadModel: OpenvinoWrapper)
  case class PaliGemmaWrappers(
      languageModel: OpenvinoWrapper,
      imageEncoder: OpenvinoWrapper,
      textEmbeddings: OpenvinoWrapper,
      modelMerger: OpenvinoWrapper)
  case class Gemma3Wrappers(
      languageModel: OpenvinoWrapper,
      imageEncoder: OpenvinoWrapper,
      textEmbeddings: OpenvinoWrapper,
      modelMerger: OpenvinoWrapper)
  case class InternVLWrappers(
      languageModel: OpenvinoWrapper,
      imageEncoder: OpenvinoWrapper,
      textEmbeddings: OpenvinoWrapper,
      modelMerger: OpenvinoWrapper)
  case class Florence2Wrappers(
      encoderModel: OpenvinoWrapper,
      decoderModel: OpenvinoWrapper,
      textEmbeddingsModel: OpenvinoWrapper,
      imageEmbedModel: OpenvinoWrapper,
      modelMergerModel: OpenvinoWrapper)
}
