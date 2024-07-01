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

package com.johnsnowlabs.ml.onnx

import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.OrtSession.SessionOptions.{ExecutionMode, OptLevel}
import ai.onnxruntime.providers.OrtCUDAProviderOptions
import ai.onnxruntime.{OrtEnvironment, OrtSession}
import com.johnsnowlabs.util.{ConfigHelper, FileHelper, ZipArchiveUtil}
import org.apache.commons.io.FileUtils
import org.slf4j.{Logger, LoggerFactory}
import org.apache.hadoop.fs.{FileSystem, Path}
import java.io._
import java.nio.file.{Files, Paths}
import java.util.UUID
import scala.util.{Failure, Success, Try}

class OnnxWrapper(var onnxModel: Array[Byte], var onnxModelPath: Option[String] = None)
    extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null, null)
  }

  // Important for serialization on none-kyro serializers
  @transient private var ortSession: OrtSession = _
  @transient private var ortEnv: OrtEnvironment = _

  def getSession(onnxSessionOptions: Map[String, String]): (OrtSession, OrtEnvironment) =
    this.synchronized {
      // TODO: After testing it works remove the Map.empty
      if (ortSession == null && ortEnv == null) {
        val (session, env) =
          OnnxWrapper.withSafeOnnxModelLoader(onnxModel, onnxSessionOptions, onnxModelPath)
        ortEnv = env
        ortSession = session
      }
      (ortSession, ortEnv)
    }

  def saveToFile(file: String, zip: Boolean = true): Unit = {
    // 1. Create tmp director
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_onnx")
      .toAbsolutePath
      .toString

    // 2. Save onnx model
    val fileName = Paths.get(file).getFileName.toString
    val onnxFile = Paths
      .get(tmpFolder, fileName)
      .toString

    FileUtils.writeByteArrayToFile(new File(onnxFile), onnxModel)
    // 4. Zip folder
    if (zip) ZipArchiveUtil.zip(tmpFolder, file)

    // 5. Remove tmp directory
    FileHelper.delete(tmpFolder)
  }

}

/** Companion object */
object OnnxWrapper {
  private[OnnxWrapper] val logger: Logger = LoggerFactory.getLogger("OnnxWrapper")

  // TODO: make sure this.synchronized is needed or it's not a bottleneck
  private def withSafeOnnxModelLoader(
      onnxModel: Array[Byte],
      sessionOptions: Map[String, String],
      onnxModelPath: Option[String] = None): (OrtSession, OrtEnvironment) =
    this.synchronized {
      val env = OrtEnvironment.getEnvironment()
      val sessionOptionsObject = if (sessionOptions.isEmpty) {
        new SessionOptions()
      } else {
        mapToSessionOptionsObject(sessionOptions)
      }
      if (onnxModelPath.isDefined) {
        val session = env.createSession(onnxModelPath.get, sessionOptionsObject)
        (session, env)
      } else {
        val session = env.createSession(onnxModel, sessionOptionsObject)
        (session, env)
      }
    }

  // TODO: the parts related to onnx_data should be refactored once we support addFile()
  def read(
      modelPath: String,
      zipped: Boolean = true,
      useBundle: Boolean = false,
      modelName: String = "model",
      dataFileSuffix: String = "_data"): OnnxWrapper = {

    // 1. Create tmp folder
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_onnx")
      .toAbsolutePath
      .toString

    // 2. Unpack archive
    val folder =
      if (zipped)
        ZipArchiveUtil.unzip(new File(modelPath), Some(tmpFolder))
      else
        modelPath

    val sessionOptions = new OnnxSession().getSessionOptions
    val onnxFile =
      if (useBundle) Paths.get(modelPath, s"$modelName.onnx").toString
      else Paths.get(folder, new File(folder).list().head).toString

    var onnxDataFile: File = null

    // see if the onnx model has a .onnx_data file
    // get parent directory of onnx file if modelPath is a file
    val parentDir = if (zipped) Paths.get(modelPath).getParent.toString else modelPath

    val onnxDataFileExist: Boolean = {
      onnxDataFile = Paths.get(parentDir, modelName + dataFileSuffix).toFile
      onnxDataFile.exists()
    }

    if (onnxDataFileExist) {
      val onnxDataFileTmp =
        Paths.get(tmpFolder, modelName + dataFileSuffix).toFile
      FileUtils.copyFile(onnxDataFile, onnxDataFileTmp)
    }

    val modelFile = new File(onnxFile)
    val modelBytes = FileUtils.readFileToByteArray(modelFile)
    var session: OrtSession = null
    var env: OrtEnvironment = null
    if (onnxDataFileExist) {
      val (_session, _env) = withSafeOnnxModelLoader(modelBytes, sessionOptions, Some(onnxFile))
      session = _session
      env = _env
    } else {
      val (_session, _env) = withSafeOnnxModelLoader(modelBytes, sessionOptions, None)
      session = _session
      env = _env

    }
    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)

    val onnxWrapper =
      if (onnxDataFileExist) new OnnxWrapper(modelBytes, Option(onnxFile))
      else new OnnxWrapper(modelBytes)
    onnxWrapper.ortSession = session
    onnxWrapper.ortEnv = env
    onnxWrapper
  }

  private def mapToSessionOptionsObject(sessionOptions: Map[String, String]): SessionOptions = {
    val providers = OrtEnvironment.getAvailableProviders
    if (providers.toArray.map(x => x.toString).contains("CUDA")) {
      mapToCUDASessionConfig(sessionOptions)
    } else mapToCPUSessionConfig(sessionOptions)
  }

  private def mapToCUDASessionConfig(sessionOptionsMap: Map[String, String]): SessionOptions = {

    logger.info("Using CUDA")
    println("Using CUDA")
    // it seems there is no easy way to use multiple GPUs
    // at least not without using multiple threads
    // TODO: add support for multiple GPUs

    val gpuDeviceId = sessionOptionsMap(ConfigHelper.onnxGpuDeviceId).toInt

    val sessionOptions = new OrtSession.SessionOptions()
    logger.info(s"ONNX session option gpuDeviceId=$gpuDeviceId")
    val cudaOpts = new OrtCUDAProviderOptions(gpuDeviceId)
    sessionOptions.addCUDA(cudaOpts)

    sessionOptions
  }

  private def mapToCPUSessionConfig(sessionOptionsMap: Map[String, String]): SessionOptions = {

    val defaultExecutionMode = ExecutionMode.SEQUENTIAL
    val defaultOptLevel = OptLevel.ALL_OPT

    def getOptLevel(optLevel: String): OptLevel = {
      Try(OptLevel.valueOf(optLevel)) match {
        case Success(value) => value
        case Failure(_) => {
          logger.warn(
            s"Error while getting OptLevel, using default value: ${defaultOptLevel.name()}")
          defaultOptLevel
        }
      }
    }

    def getExecutionMode(executionMode: String): ExecutionMode = {
      Try(ExecutionMode.valueOf(executionMode)) match {
        case Success(value) => value
        case Failure(_) => {
          logger.warn(
            s"Error while getting Execution Mode, using default value: ${defaultExecutionMode.name()}")
          defaultExecutionMode
        }
      }
    }

    logger.info("Using CPUs")
    println("Using CPUs")
    // TODO: the following configs can be tested for performance
    // However, so far, they seem to be slower than the ones used
    // opts.setIntraOpNumThreads(Runtime.getRuntime.availableProcessors())
    // opts.setMemoryPatternOptimization(true)
    // opts.setCPUArenaAllocator(false)

    val intraOpNumThreads = sessionOptionsMap(ConfigHelper.onnxIntraOpNumThreads).toInt
    val optimizationLevel = getOptLevel(sessionOptionsMap(ConfigHelper.onnxOptimizationLevel))
    val executionMode = getExecutionMode(sessionOptionsMap(ConfigHelper.onnxExecutionMode))

    val sessionOptions = new OrtSession.SessionOptions()
    logger.info(s"ONNX session option intraOpNumThreads=$intraOpNumThreads")
    sessionOptions.setIntraOpNumThreads(intraOpNumThreads)
    logger.info(s"ONNX session option optimizationLevel=$optimizationLevel")
    sessionOptions.setOptimizationLevel(optimizationLevel)
    logger.info(s"ONNX session option executionMode=$executionMode")
    sessionOptions.setExecutionMode(executionMode)

    sessionOptions
  }

  case class EncoderDecoderWrappers(
      encoder: OnnxWrapper,
      decoder: OnnxWrapper,
      decoderWithPast: OnnxWrapper)

  case class DecoderWrappers(decoder: OnnxWrapper)

  case class EncoderDecoderWithoutPastWrappers(encoder: OnnxWrapper, decoder: OnnxWrapper)

}
