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
import com.johnsnowlabs.util.{FileHelper, ZipArchiveUtil}
import org.apache.commons.io.FileUtils
import org.slf4j.{Logger, LoggerFactory}

import java.io._
import java.nio.file.{Files, Paths}
import java.util.UUID

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
    val onnxFile = Paths.get(tmpFolder, file).toString
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

      val opts =
        if (sessionOptions.isDefined) sessionOptions.get else new OrtSession.SessionOptions()

      val providers = OrtEnvironment.getAvailableProviders

      if (providers.toArray.map(x => x.toString).contains("CUDA")) {
        logger.info("using CUDA")
        // it seems there is no easy way to use multiple GPUs
        // at least not without using multiple threads
        // TODO: add support for multiple GPUs
        // TODO: allow user to specify which GPU to use
        val gpuDeviceId = 0 // The GPU device ID to execute on
        val cudaOpts = new OrtCUDAProviderOptions(gpuDeviceId)
        // TODO: incorporate other cuda-related configs
        // cudaOpts.add("gpu_mem_limit", "" + (512 * 1024 * 1024))
        // sessOptions.addCUDA(gpuDeviceId)
        opts.addCUDA(cudaOpts)
      } else {
        logger.info("using CPUs")
        // TODO: the following configs can be tested for performance
        // However, so far, they seem to be slower than the ones used
        // opts.setIntraOpNumThreads(Runtime.getRuntime.availableProcessors())
        // opts.setMemoryPatternOptimization(true)
        // opts.setCPUArenaAllocator(false)
        opts.setIntraOpNumThreads(6)
        opts.setOptimizationLevel(OptLevel.ALL_OPT)
        opts.setExecutionMode(ExecutionMode.SEQUENTIAL)
      }

      val session = env.createSession(onnxModel, opts)
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

  case class EncoderDecoderWrappers(
      encoder: OnnxWrapper,
      decoder: OnnxWrapper,
      decoderWithPast: OnnxWrapper)

}
