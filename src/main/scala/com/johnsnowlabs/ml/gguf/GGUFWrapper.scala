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

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import de.kherud.llama.{LlamaModel, ModelParameters}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkFiles
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.SparkSession
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import java.nio.file.Paths

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
          modelParameters.setModel(modelFilePath)
          llamaModel = GGUFWrapper.withSafeGGUFModelLoader(modelParameters)
        } else
          throw new IllegalStateException(
            s"Model file $modelFileName does not exist in SparkFiles.")
      }
      // TODO: if the model is already loaded then the model parameters will not apply. perhaps output a logline here.
      llamaModel
    }

  def saveToFile(path: String): Unit = {
    val fileSystem: FileSystem = ResourceHelper.fileSystemFromPath(path)

    val modelFilePath = new Path(SparkFiles.get(modelFileName))
    fileSystem.copyFromLocalFile(modelFilePath, new Path(path))
  }

  // Destructor to free the model when this object is garbage collected
  override def finalize(): Unit = close()

  /** Closes the underlying LlamaModel and frees resources. */
  def close(): Unit = {
    this.synchronized {
      if (llamaModel != null) {
        println("Closing llama.cpp model.")
        llamaModel.close()
        llamaModel = null
      }
    }
  }

}

/** Companion object */
object GGUFWrapper {
  private[GGUFWrapper] val logger: Logger = LoggerFactory.getLogger("GGUFWrapper")

  // TODO: make sure this.synchronized is needed or it's not a bottleneck
  private def withSafeGGUFModelLoader(modelParameters: ModelParameters): LlamaModel =
    this.synchronized {
      new LlamaModel(modelParameters)
    }

  /** Reads the GGUF model from file during loadSavedModel. */
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

  /** Finds the GGUF model file in the given folder, returning its absolute path.
    *
    * @param folderPath
    *   Path to the folder containing the GGUF model file.
    * @return
    *   The absolute path to the GGUF model file.
    */
  def findGGUFModelInFolder(folderPath: String): String = {
    val folder = new File(folderPath)
    if (folder.exists && folder.isDirectory) {
      val ggufFile: String = folder.listFiles
        .find(f =>
          // We only find the first file, that is not mmproj (in case this was originally a multi-modal model)
          f.isFile && f.getName
            .endsWith(".gguf") && !f.getName.toLowerCase().contains("mmproj")) match {
        case Some(ggufFile) => ggufFile.getAbsolutePath
        case None =>
          throw new IllegalArgumentException(s"Could not find GGUF model in $folderPath")
      }

      new File(ggufFile).getAbsolutePath
    } else {
      throw new IllegalArgumentException(s"Path $folderPath is not a directory")
    }
  }

  /** Reads the GGUF model from the folder passed by the Spark Reader during loading of a
    * serialized model.
    */
  def readModel(modelFolderPath: String, spark: SparkSession): GGUFWrapper = {
    // In case the path belongs to a different file system but doesn't have the scheme prepended (e.g. dbfs)
    val fileSystem: FileSystem = ResourceHelper.fileSystemFromPath(modelFolderPath)
    val actualFolderPath = fileSystem.resolvePath(new Path(modelFolderPath)).toString
    val localFolder = ResourceHelper.copyToLocal(actualFolderPath)
    val modelFile = findGGUFModelInFolder(localFolder)
    read(spark, modelFile)
  }

  /** Closes the broadcasted GGUFWrapper model on all Spark workers and the driver, freeing up
    * resources.
    *
    * We use a foreachPartition on a dummy RDD to ensure that the close method is called on each
    * executor.
    *
    * @param broadcastedModel
    *   An optional Broadcast[GGUFWrapper] instance to be closed. If None, no action is taken.
    */
  def closeBroadcastModel(broadcastedModel: Option[Broadcast[GGUFWrapper]]): Unit = {
    def closeOnWorkers(): Unit = {
      val spark = SparkSession.getActiveSession.get
      // Get the number of executors to ensure we run a task on each one.
      val numExecutors = spark.sparkContext.getExecutorMemoryStatus.size

      // Create a dummy RDD with one partition per executor
      val dummyRdd = spark.sparkContext.parallelize(1 to numExecutors, numExecutors)

      // Run a job whose only purpose is to trigger the shutdown method on each worker
      dummyRdd.foreachPartition { _ =>
        broadcastedModel match {
          case Some(broadcastModel) =>
            broadcastModel.value.close()
          case None => // No model to close
        }
      }
    }

    closeOnWorkers()
    broadcastedModel match {
      case Some(broadcastModel) =>
        broadcastModel.value.close() // Close the model on the driver as well
        broadcastModel.unpersist()
      case None => // No model to close
    }
  }
}
