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
