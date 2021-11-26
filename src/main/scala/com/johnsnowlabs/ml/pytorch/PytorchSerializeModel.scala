package com.johnsnowlabs.ml.pytorch

import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

import java.io.File
import java.nio.file.Files
import java.util.UUID

trait WritePytorchModel {

  def writePytorchModel(path: String, spark: SparkSession, pytorchWrapper: PytorchWrapper, fileName: String): Unit = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    val tmpFolder = pytorchWrapper.saveToFile(fileName)

    //1. Copy to dest folder
    fileSystem.copyFromLocalFile(new Path(tmpFolder + "/" + fileName), new Path(path))

    //2. Remove tmp folder
    FileUtils.deleteDirectory(new File(tmpFolder))

  }

}

trait ReadPytorchModel {

  val torchscriptFile: String

  def readPytorchModel(path: String, spark: SparkSession, suffix: String): PytorchWrapper = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val localFileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp directory
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + suffix)
      .toAbsolutePath.toString

    // 2. Copy to local dir
    localFileSystem.copyToLocalFile(new Path(path), new Path(tmpFolder))

    val localPath = new Path(tmpFolder, torchscriptFile).toString
    val pytorchWrapper = PytorchWrapper(localPath, "local")
    pytorchWrapper
  }

}
