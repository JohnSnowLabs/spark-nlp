/*
 * Copyright 2017-2023 John Snow Labs
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

import ai.onnxruntime.OrtEnvironment
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.slf4j.{Logger, LoggerFactory}

import java.io.Serializable

class OnnxSession extends Serializable {

  // Important for serialization on none-kyro serializers
  @transient val logger: Logger = LoggerFactory.getLogger("OnnxSession")

  def getSessionOptions: Map[String, String] = {
    val providers = OrtEnvironment.getAvailableProviders
    if (providers.toArray.map(x => x.toString).contains("CUDA")) {
      getCUDASessionConfig
    } else getCPUSessionConfig
  }

  private def getCUDASessionConfig: Map[String, String] = {
    val gpuDeviceId = ConfigLoader.getConfigIntValue(ConfigHelper.onnxGpuDeviceId)
    Map(ConfigHelper.onnxGpuDeviceId -> gpuDeviceId.toString)
  }

  private def getCPUSessionConfig: Map[String, String] = {
    val intraOpNumThreads =
      ConfigLoader.getConfigIntValue(ConfigHelper.onnxIntraOpNumThreads)
    val optimizationLevel =
      ConfigLoader.getConfigStringValue(ConfigHelper.onnxOptimizationLevel)
    val executionMode =
      ConfigLoader.getConfigStringValue(ConfigHelper.onnxExecutionMode)

    Map(ConfigHelper.onnxIntraOpNumThreads -> intraOpNumThreads.toString) ++
      Map(ConfigHelper.onnxOptimizationLevel -> optimizationLevel) ++
      Map(ConfigHelper.onnxExecutionMode -> executionMode)
  }

}
