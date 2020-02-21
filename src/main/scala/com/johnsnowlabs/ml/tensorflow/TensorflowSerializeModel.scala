package com.johnsnowlabs.ml.tensorflow

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID

import com.johnsnowlabs.nlp.annotators.ner.dl.LoadsContrib
import com.johnsnowlabs.util.FileHelper
import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

trait WriteTensorflowModel {


  def writeTensorflowModel(
                            path: String,
                            spark: SparkSession,
                            tensorflow: TensorflowWrapper,
                            suffix: String, filename:String,
                            configProtoBytes: Option[Array[Byte]] = None
                          ): Unit = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp folder
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + suffix)
      .toAbsolutePath.toString

    val tfFile = Paths.get(tmpFolder, filename).toString

    // 2. Save Tensorflow state
    tensorflow.saveToFile(tfFile, configProtoBytes)

    // 3. Copy to dest folder
    fs.copyFromLocalFile(new Path(tfFile), new Path(path))

    // 4. Remove tmp folder
    FileUtils.deleteDirectory(new File(tmpFolder))
  }

  def writeTensorflowHub(
                          path: String,
                          tfPath: String,
                          spark: SparkSession,
                          suffix: String = "_use"
                        ): Unit = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp folder
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + suffix)
      .toAbsolutePath.toString

    // 2. Get the paths to saved_model.pb and variables directory
    val savedModelPath = Paths.get(tfPath, "saved_model.pb").toString
    val variableFilesPath = Paths.get(tfPath, "variables").toString

    // 3. Copy to dest folder
    fs.copyFromLocalFile(new Path(savedModelPath), new Path(path))
    fs.copyFromLocalFile(new Path(variableFilesPath), new Path(path))

    // 4. Remove tmp folder
    FileUtils.deleteDirectory(new File(tmpFolder))
  }

}

trait ReadTensorflowModel {
  val tfFile: String

  def readTensorflowModel(
                           path: String,
                           spark: SparkSession,
                           suffix: String,
                           zipped:Boolean = true,
                           useBundle:Boolean = false,
                           tags:Array[String]=Array.empty,
                           initAllTables: Boolean = false
                         ): TensorflowWrapper = {

    LoadsContrib.loadContribToCluster(spark)

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp directory
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12)+ suffix)
      .toAbsolutePath.toString

    // 2. Copy to local dir
    fs.copyToLocalFile(new Path(path, tfFile), new Path(tmpFolder))

    // 3. Read Tensorflow state
    val tf = TensorflowWrapper.read(new Path(tmpFolder, tfFile).toString,
      zipped, tags = tags, useBundle = useBundle, initAllTables = initAllTables)

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)

    tf
  }

  def readTensorflowHub(
                         path: String,
                         spark: SparkSession,
                         suffix: String,
                         zipped:Boolean = true,
                         useBundle:Boolean = false,
                         tags:Array[String]=Array.empty
                       ): TensorflowWrapper = {


    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp directory
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12)+ suffix)
      .toAbsolutePath.toString

    // 2. Copy to local dir
    fs.copyToLocalFile(new Path(path, "saved_model.pb"), new Path(tmpFolder))
    fs.copyToLocalFile(new Path(path, "variables"), new Path(tmpFolder))

    // 3. Read Tensorflow state
    val tf = TensorflowWrapper.read(new Path(tmpFolder).toString,
      zipped, tags = tags, useBundle = useBundle)

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)

    tf
  }
}
