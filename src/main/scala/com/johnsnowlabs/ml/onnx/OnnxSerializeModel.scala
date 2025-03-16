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

import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID

trait WriteOnnxModel {

  def writeOnnxModels(
      path: String,
      spark: SparkSession,
      onnxWrappersWithNames: Seq[(OnnxWrapper, String)],
      suffix: String): Unit = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp folder
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + suffix)
      .toAbsolutePath
      .toString

    onnxWrappersWithNames foreach { case (onnxWrapper, modelName) =>
      val onnxFile = Paths.get(tmpFolder, modelName).toString

      // 2. Save ONNX state
      onnxWrapper.saveToFile(onnxFile)

      // 3. Copy to dest folder
      fileSystem.copyFromLocalFile(new Path(onnxFile), new Path(path))

      // 4. check if there is a onnx_data file
      if (onnxWrapper.dataFileDirectory.isDefined) {
        val onnxDataFile = new Path(onnxWrapper.dataFileDirectory.get)
        if (fileSystem.exists(onnxDataFile)) {
          fileSystem.copyFromLocalFile(onnxDataFile, new Path(path))
        }
      }

    }

    // 4. Remove tmp folder
    FileUtils.deleteDirectory(new File(tmpFolder))
  }

  def writeOnnxModel(
      path: String,
      spark: SparkSession,
      onnxWrapper: OnnxWrapper,
      suffix: String,
      fileName: String): Unit = {
    writeOnnxModels(path, spark, Seq((onnxWrapper, fileName)), suffix)
  }
}

trait ReadOnnxModel {
  val onnxFile: String

  def readOnnxModel(
      path: String,
      spark: SparkSession,
      suffix: String,
      zipped: Boolean = true,
      useBundle: Boolean = false,
      modelName: Option[String] = None,
      tmpFolder: Option[String] = None,
      dataFilePostfix: Option[String] = None): OnnxWrapper = {

    // 1. Copy to local tmp dir
    val localModelFile = if (modelName.isDefined) modelName.get else onnxFile
    val srcPath = new Path(path, localModelFile)
    val fileSystem = getFileSystem(path, spark)
    val localTmpFolder = if (tmpFolder.isDefined) tmpFolder.get else createTmpDirectory(suffix)
    fileSystem.copyToLocalFile(srcPath, new Path(localTmpFolder))

    // 2. Copy onnx_data file if exists
    val fsPath = new Path(path, localModelFile).toString

    val onnxDataFile: Option[String] = if (modelName.isDefined && dataFilePostfix.isDefined) {
      var modelNameWithoutSuffix = modelName.get.replace(".onnx", "")
      Some(
        fsPath.replaceAll(
          modelName.get,
          s"${suffix}_${modelNameWithoutSuffix}${dataFilePostfix.get}"))
    } else None

    if (onnxDataFile.isDefined) {
      val onnxDataFilePath = new Path(onnxDataFile.get)
      if (fileSystem.exists(onnxDataFilePath)) {
        fileSystem.copyToLocalFile(onnxDataFilePath, new Path(localTmpFolder))
      }
    }

    // 3. Read ONNX state
    val onnxFileTmpPath = new Path(localTmpFolder, localModelFile).toString
    val onnxWrapper =
      OnnxWrapper.read(
        spark,
        onnxFileTmpPath,
        zipped = zipped,
        useBundle = useBundle,
        modelName = if (modelName.isDefined) modelName.get else onnxFile,
        onnxFileSuffix = Some(suffix),
        dataFileSuffix = dataFilePostfix)

    onnxWrapper

  }

  private def getFileSystem(path: String, sparkSession: SparkSession): FileSystem = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fileSystem = FileSystem.get(uri, sparkSession.sparkContext.hadoopConfiguration)
    fileSystem
  }

  private def createTmpDirectory(suffix: String): String = {

    // 1. Create tmp directory
    val tmpFolder = Files
      .createTempDirectory(s"${UUID.randomUUID().toString.takeRight(12)}_$suffix")
      .toAbsolutePath
      .toString

    tmpFolder
  }

  def readOnnxModels(
      path: String,
      spark: SparkSession,
      modelNames: Seq[String],
      suffix: String,
      zipped: Boolean = true,
      useBundle: Boolean = false,
      dataFilePostfix: String = "_data"): Map[String, OnnxWrapper] = {

    val tmpFolder = Some(createTmpDirectory(suffix))

    val wrappers = (modelNames map { modelName: String =>
      val onnxWrapper = readOnnxModel(
        path,
        spark,
        suffix,
        zipped,
        useBundle,
        Some(modelName),
        tmpFolder,
        Option(dataFilePostfix))
      (modelName, onnxWrapper)
    }).toMap

    wrappers
  }

}
