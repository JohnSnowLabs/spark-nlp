package com.johnsnowlabs.nlp.annotators.assertion.dl

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID

import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.nlp.annotators.spell.ocr.OcrSpellCheckModel
import com.johnsnowlabs.util.FileHelper
import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

/**
  * Created by jose on 23/03/18.
  */
trait WriteTensorflowModel{


  def writeTensorflowModel(path: String, spark: SparkSession, tensorflow: TensorflowWrapper, suffix: String): Unit = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp folder
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + suffix)
      .toAbsolutePath.toString
    // TODO fix this hardcoded tfFile stuff
    val tfFile = Paths.get(tmpFolder, OcrSpellCheckModel.tfFile).toString

    // 2. Save Tensorflow state
    tensorflow.saveToFile(tfFile)

    // 3. Copy to dest folder
    fs.copyFromLocalFile(new Path(tfFile), new Path(path))

    // 4. Remove tmp folder
    FileUtils.deleteDirectory(new File(tmpFolder))
  }

}

trait ReadTensorflowModel {
  val tfFile: String

  def readTensorflowModel(path: String, spark: SparkSession, suffix: String, zipped:Boolean = true): TensorflowWrapper = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp directory
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12)+ suffix)
      .toAbsolutePath.toString

    // 2. Copy to local dir
    fs.copyToLocalFile(new Path(path, tfFile), new Path(tmpFolder))

    // TODO remove hardcoded useBundle, tags
    // 3. Read Tensorflow state
    val tf = TensorflowWrapper.read(new Path(tmpFolder, tfFile).toString, zipped, tags = Array("our-graph"), useBundle = true)

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)

    tf
  }
}
