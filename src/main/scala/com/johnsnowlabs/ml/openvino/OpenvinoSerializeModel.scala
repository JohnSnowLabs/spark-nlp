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

import com.johnsnowlabs.util.FileHelper
import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID

trait WriteOpenvinoModel {

  def writeOpenvinoModels(
      path: String,
      spark: SparkSession,
      ovWrappersWithNames: Seq[(OpenvinoWrapper, String)],
      suffix: String): Unit = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp folder
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + suffix)
      .toAbsolutePath
      .toString

    ovWrappersWithNames foreach { case (ovWrapper, modelName) =>
      val savedOvModel = Paths.get(tmpFolder, modelName).toString
      ovWrapper.saveToFile(savedOvModel)
      fileSystem.copyFromLocalFile(new Path(savedOvModel), new Path(path))
    }

    // 4. Remove tmp folder
    FileUtils.deleteDirectory(new File(tmpFolder))
  }

  def writeOpenvinoModel(
      path: String,
      spark: SparkSession,
      openvinoWrapper: OpenvinoWrapper,
      suffix: String,
      fileName: String): Unit = {
    writeOpenvinoModels(path, spark, Seq((openvinoWrapper, fileName)), suffix)
  }
}

trait ReadOpenvinoModel {
  val openvinoFile: String

  def readOpenvinoModel(
      path: String,
      spark: SparkSession,
      suffix: String,
      zipped: Boolean = true): OpenvinoWrapper = {
    val ovModel = readOpenvinoModels(path, spark, Seq(openvinoFile), suffix, zipped)
    ovModel(openvinoFile)
  }

  def readOpenvinoModels(
      path: String,
      spark: SparkSession,
      modelNames: Seq[String],
      suffix: String,
      zipped: Boolean = true): Map[String, OpenvinoWrapper] = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp directory
    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + suffix)
      .toAbsolutePath
      .toString

    val wrappers = (modelNames map { modelName: String =>
      // 2. Copy to local dir
      val srcPath = new Path(path, modelName)
      fileSystem.copyToLocalFile(srcPath, new Path(tmpFolder))
      val localPath = new Path(tmpFolder, modelName).toString

      val ovWrapper =
        OpenvinoWrapper.read(
          spark,
          localPath,
          zipped = zipped,
          modelName = modelName,
          ovFileSuffix = Some(suffix))
      (modelName, ovWrapper)
    }).toMap

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)

    wrappers
  }
}
