/*
 * Copyright 2017-2024 John Snow Labs
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
package com.johnsnowlabs.ml.gguf

import com.johnsnowlabs.nlp.llama.{LlamaModel, ModelParameters}
import org.apache.spark.SparkFiles
import org.apache.spark.sql.SparkSession
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import java.nio.file.{Files, Paths}

class GGUFWrapper(var modelFileName: String, var modelFolder: String) extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null, null)
  }

  // Important for serialization on none-kryo serializers
  @transient private var llamaModel: LlamaModel = _

  def getSession(modelParameters: ModelParameters): LlamaModel =
    this.synchronized {
      if (llamaModel == null) {
        // TODO: Validate when modelFileName or tmpFolder is None??
        val modelFilePath = SparkFiles.get(modelFileName)

        if (Paths.get(modelFilePath).toFile.exists()) {
          modelParameters.setModelFilePath(modelFilePath)
          llamaModel = GGUFWrapper.withSafeGGUFModelLoader(modelParameters)
        } else
          throw new IllegalStateException(
            s"Model file $modelFileName does not exist in SparkFiles.")
      }
      // TODO: if the model is already loaded then the model parameters will not apply. perhaps output a logline here.
      llamaModel
    }

  def saveToFile(file: String): Unit = {
    val modelFilePath = SparkFiles.get(modelFileName)
    val modelOutputPath = Paths.get(file, modelFileName)
    Files.copy(Paths.get(modelFilePath), modelOutputPath)
  }

  // Destructor to free the model when this object is garbage collected
  override def finalize(): Unit = {
    if (llamaModel != null) {
      llamaModel.close()
    }
  }

}

/** Companion object */
object GGUFWrapper {
  private[GGUFWrapper] val logger: Logger = LoggerFactory.getLogger("GGUFWrapper")

  // TODO: make sure this.synchronized is needed or it's not a bottleneck
  private def withSafeGGUFModelLoader(modelParameters: ModelParameters): LlamaModel =
    this.synchronized {
      new LlamaModel(modelParameters) // TODO: Model parameters
    }

  def read(sparkSession: SparkSession, modelPath: String): GGUFWrapper = {
    // TODO Better Sanity Check
    val modelFile = new File(modelPath)
    val modelFileExist: Boolean = modelFile.exists()

    if (!modelFile.getName.endsWith(".gguf"))
      throw new IllegalArgumentException(s"Model file $modelPath is not a GGUF model file")

    if (modelFileExist) {
      sparkSession.sparkContext.addFile(modelPath)
    } else throw new IllegalArgumentException(s"Model file $modelPath does not exist")

    new GGUFWrapper(modelFile.getName, modelFile.getParent)
  }
}
