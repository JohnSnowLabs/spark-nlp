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

import com.johnsnowlabs.nlp.llama.ModelParametersMultiModal
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import de.kherud.llama.{LlamaModel, ModelParameters}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkFiles
import org.apache.spark.sql.SparkSession

import java.io.File
import java.nio.file.{Files, Paths}

class GGUFWrapperMultiModal(var modelFileName: String, var mmprojFileName: String)
    extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null, null)
  }

  // Important for serialization on none-kryo serializers
  @transient private var llamaModel: LlamaModel = _

  def getSession(modelParameters: ModelParameters): LlamaModel =
    this.synchronized {
      if (llamaModel == null) {
        val modelFilePath = SparkFiles.get(modelFileName)
        val mmprojFilePath = SparkFiles.get(mmprojFileName)
        val filesExist =
          Paths.get(modelFilePath).toFile.exists() && Paths.get(mmprojFilePath).toFile.exists()

        if (filesExist) {
          modelParameters.setModel(modelFilePath)
          val mparamsMultiModal = new ModelParametersMultiModal(modelParameters, mmprojFilePath)
          llamaModel = GGUFWrapperMultiModal.withSafeGGUFModelLoader(mparamsMultiModal)
        } else
          throw new IllegalStateException(
            s"Model file $modelFileName does not exist in SparkFiles.")
      }
      // TODO: if the model is already loaded then the model parameters will not apply. perhaps output a logline here.
      llamaModel
    }

  def saveToFile(path: String): Unit = {
    val fileSystem = ResourceHelper.fileSystemFromPath(path)
    val modelFilePath = new Path(SparkFiles.get(modelFileName))
    val mmprojFilePath = new Path(SparkFiles.get(mmprojFileName))
    val outPath = new Path(path)
    fileSystem.copyFromLocalFile(modelFilePath, outPath)
    fileSystem.copyFromLocalFile(mmprojFilePath, outPath)
  }

  // Destructor to free the model when this object is garbage collected
  override def finalize(): Unit = {
    if (llamaModel != null) {
      llamaModel.close()
    }
  }

}

/** Companion object */
object GGUFWrapperMultiModal {
  private def withSafeGGUFModelLoader(modelParameters: ModelParametersMultiModal): LlamaModel =
    this.synchronized {
      new LlamaModel(modelParameters)
    }

  /** Reads the GGUF model from file during loadSavedModel. */
  def read(
      sparkSession: SparkSession,
      modelPath: String,
      mmprojPath: String): GGUFWrapperMultiModal = {
    val modelFile = new File(modelPath)
    val mmprojFile = new File(mmprojPath)

    if (!modelFile.getName.endsWith(".gguf"))
      throw new IllegalArgumentException(s"Model file $modelPath is not a GGUF model file")

    if (!mmprojFile.getName.endsWith(".gguf"))
      throw new IllegalArgumentException(s"mmproj file $mmprojPath is not a GGUF model file")

    if (!mmprojFile.getName.contains("mmproj"))
      throw new IllegalArgumentException(
        s"mmproj file $mmprojPath is not a GGUF mmproj file (should contain 'mmproj' in its name)")

    if (modelFile.exists() && mmprojFile.exists()) {
      sparkSession.sparkContext.addFile(modelPath)
      sparkSession.sparkContext.addFile(mmprojPath)
    } else
      throw new IllegalArgumentException(
        s"Model file $modelPath or mmproj file $mmprojPath does not exist")

    new GGUFWrapperMultiModal(modelFile.getName, mmprojFile.getName)
  }

  def findGGUFModelsInFolder(folderPath: String): (String, String) = {
    val folder = new File(folderPath)
    if (folder.exists && folder.isDirectory) {
      val ggufFiles: Array[String] = folder.listFiles
        .filter(_.isFile)
        .filter(_.getName.endsWith(".gguf"))
        .map(_.getAbsolutePath)

      val (ggufMainPath, ggufMmprojPath) =
        if (ggufFiles.length == 2 && ggufFiles.exists(_.contains("mmproj"))) {
          val Array(firstModel, secondModel) = ggufFiles
          if (firstModel.contains("mmproj")) (secondModel, firstModel)
          else (firstModel, secondModel)
        } else
          throw new IllegalArgumentException(
            s"Could not determine main GGUF model or mmproj GGUF model in $folderPath." +
              s" The folder should contain exactly two files:" +
              s" One main GGUF model and one mmproj GGUF model." +
              s" The mmproj model should have 'mmproj' in its name.")

      (ggufMainPath, ggufMmprojPath)
    } else {
      throw new IllegalArgumentException(s"Path $folderPath is not a directory")
    }
  }

  /** Reads the GGUF model from the folder passed by the Spark Reader during loading of a
    * serialized model.
    */
  def readModel(modelFolderPath: String, spark: SparkSession): GGUFWrapperMultiModal = {
    val uri = new java.net.URI(modelFolderPath.replaceAllLiterally("\\", "/"))
    // In case the path belongs to a different file system but doesn't have the scheme prepended (e.g. dbfs)
    val fileSystem: FileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val actualFolderPath = fileSystem.resolvePath(new Path(modelFolderPath)).toString
    val localFolder = ResourceHelper.copyToLocal(actualFolderPath)
    val (ggufMainPath, ggufMmprojPath) = findGGUFModelsInFolder(localFolder)
    read(spark, ggufMainPath, ggufMmprojPath)
  }
}
