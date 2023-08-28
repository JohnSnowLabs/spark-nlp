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

  def writeOpenvinoModel(
      path: String,
      spark: SparkSession,
      openvinoWrapper: OpenvinoWrapper,
      suffix: String,
      fileName: String): Unit = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + suffix)
      .toAbsolutePath
      .toString

    val savedOvModel = Paths.get(tmpFolder, fileName).toString
    openvinoWrapper.saveToFile(savedOvModel)

    fs.copyFromLocalFile(new Path(savedOvModel), new Path(path))
    FileUtils.deleteDirectory(new File(tmpFolder))
  }
}

trait ReadOpenvinoModel {
  val openvinoFile: String

  def readOpenvinoModel(
      path: String,
      spark: SparkSession,
      suffix: String,
      zipped: Boolean = true): OpenvinoWrapper = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    val tmpFolder = Files
      .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + suffix)
      .toAbsolutePath
      .toString

    fs.copyToLocalFile(new Path(path, openvinoFile), new Path(tmpFolder))
    val localPath = new Path(tmpFolder, openvinoFile).toString
    val (openvinoWrapper, _) = OpenvinoWrapper.fromOpenvinoFormat(localPath, zipped = zipped)

    FileHelper.delete(tmpFolder)
    openvinoWrapper
  }
}
