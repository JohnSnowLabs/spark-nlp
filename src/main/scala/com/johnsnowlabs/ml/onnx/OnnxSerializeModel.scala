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
import com.johnsnowlabs.util.FileHelper
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
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

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
      fs.copyFromLocalFile(new Path(onnxFile), new Path(path))
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
      sessionOptions: Option[SessionOptions] = None): OnnxWrapper = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp directory
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + suffix)
      .toAbsolutePath
      .toString

    // 2. Copy to local dir
    fs.copyToLocalFile(new Path(path, onnxFile), new Path(tmpFolder))

    val localPath = new Path(tmpFolder, onnxFile).toString

    // 3. Read ONNX state
    val onnxWrapper = OnnxWrapper.read(
      localPath,
      zipped = zipped,
      useBundle = useBundle,
      sessionOptions = sessionOptions)

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)

    onnxWrapper
  }

  def readOnnxModels(
      path: String,
      spark: SparkSession,
      modelNames: Seq[String],
      suffix: String,
      zipped: Boolean = true,
      useBundle: Boolean = false,
      sessionOptions: Option[SessionOptions] = None): Map[String, OnnxWrapper] = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp directory
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + suffix)
      .toAbsolutePath
      .toString

    val wrappers = (modelNames map { modelName: String =>
      // 2. Copy to local dir
      val localModelFile = modelName
      fs.copyToLocalFile(new Path(path, localModelFile), new Path(tmpFolder))

      val localPath = new Path(tmpFolder, localModelFile).toString

      // 3. Read ONNX state
      val onnxWrapper = OnnxWrapper.read(
        localPath,
        zipped = zipped,
        useBundle = useBundle,
        sessionOptions = sessionOptions)

      (modelName, onnxWrapper)
    }).toMap

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)

    wrappers
  }

}
