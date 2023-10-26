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
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader, FileHelper, ZipArchiveUtil}
import org.apache.commons.io.FileUtils
import org.slf4j.{Logger, LoggerFactory}

import java.io._
import java.nio.file.{Files, Paths}
import java.util.UUID
import scala.util.{Failure, Success, Try}

class OnnxWrapper(var onnxModel: Array[Byte]) extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null)
  }

  // Important for serialization on none-kyro serializers
  @transient private var m_session: OrtSession = _
  @transient private var m_env: OrtEnvironment = _
  @transient private val logger = LoggerFactory.getLogger("OnnxWrapper")

  def getSession(sessionOptions: Option[SessionOptions] = None): (OrtSession, OrtEnvironment) =
    this.synchronized {
      if (m_session == null && m_env == null) {
        val (session, env) = OnnxWrapper.withSafeOnnxModelLoader(onnxModel, sessionOptions)
        m_env = env
        m_session = session
      }
      (m_session, m_env)
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
      sessionOptions: Option[SessionOptions] = None): (OrtSession, OrtEnvironment) =
    this.synchronized {
      val env = OrtEnvironment.getEnvironment()
      val providers = OrtEnvironment.getAvailableProviders

      val sessionOptionsConfig = if (providers.toArray.map(x => x.toString).contains("CUDA")) {
        getCUDASessionConfig
      } else {
        getCPUSessionConfig
      }

      val session = env.createSession(onnxModel, sessionOptionsConfig)
      (session, env)
    }

  def read(
      modelPath: String,
      zipped: Boolean = true,
      useBundle: Boolean = false,
      modelName: String = "model",
      sessionOptions: Option[SessionOptions] = None): OnnxWrapper = {

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

    // TODO: simplify this logic of useBundle
    val (session, env, modelBytes) =
      if (useBundle) {
        val onnxFile = Paths.get(modelPath, s"$modelName.onnx").toString
        val modelFile = new File(onnxFile)
        val modelBytes = FileUtils.readFileToByteArray(modelFile)
        val (session, env) = withSafeOnnxModelLoader(modelBytes, sessionOptions)
        (session, env, modelBytes)
      } else {
        val modelFile = new File(folder).list().head
        val fullPath = Paths.get(folder, modelFile).toFile
        val modelBytes = FileUtils.readFileToByteArray(fullPath)
        val (session, env) = withSafeOnnxModelLoader(modelBytes, sessionOptions)
        (session, env, modelBytes)
      }

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)

    val onnxWrapper = new OnnxWrapper(modelBytes)
    onnxWrapper.m_session = session
    onnxWrapper.m_env = env
    onnxWrapper
  }

  private def getCUDASessionConfig: SessionOptions = {

    logger.info("Using CUDA")
    // it seems there is no easy way to use multiple GPUs
    // at least not without using multiple threads
    // TODO: add support for multiple GPUs

    val gpuDeviceId = ConfigLoader.getConfigIntValue(ConfigHelper.onnxGpuDeviceId)

    val sessionOptions = new OrtSession.SessionOptions()
    logger.info(s"ONNX session option gpuDeviceId=$gpuDeviceId")
    val cudaOpts = new OrtCUDAProviderOptions(gpuDeviceId)
    sessionOptions.addCUDA(cudaOpts)

    sessionOptions
  }

  private def getCPUSessionConfig: SessionOptions = {

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
    // TODO: the following configs can be tested for performance
    // However, so far, they seem to be slower than the ones used
    // opts.setIntraOpNumThreads(Runtime.getRuntime.availableProcessors())
    // opts.setMemoryPatternOptimization(true)
    // opts.setCPUArenaAllocator(false)

    val intraOpNumThreads = ConfigLoader.getConfigIntValue(ConfigHelper.onnxIntraOpNumThreads)
    val optimizationLevel = getOptLevel(
      ConfigLoader.getConfigStringValue(ConfigHelper.onnxOptimizationLevel))
    val executionMode = getExecutionMode(
      ConfigLoader.getConfigStringValue(ConfigHelper.onnxExecutionMode))

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
}
