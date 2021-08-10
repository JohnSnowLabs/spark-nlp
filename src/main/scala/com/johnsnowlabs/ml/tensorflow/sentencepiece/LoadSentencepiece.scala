package com.johnsnowlabs.ml.tensorflow.sentencepiece

import java.io.{BufferedOutputStream, File, FileOutputStream}
import java.nio.file.{Files, Paths}
import java.util.UUID

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.commons.lang.SystemUtils
import org.apache.spark.SparkFiles
import org.apache.spark.sql.SparkSession
import org.tensorflow.TensorFlow

object LoadSentencepiece {
  @transient var loadedToCluster = false
  @transient var loadedToTensorflow = false

  private lazy val lib = "_sentencepiece_tokenizer"

  private def resourcePath(os: String, lib: String) = "sp-process/"+os+"/"+lib

  lazy val sentencepiecePaths: Option[String] =
    if (SystemUtils.IS_OS_MAC || SystemUtils.IS_OS_MAC_OSX) {
      val libWithExtension = lib + ".dylib"
      Some(resourcePath("mac", libWithExtension))
    } else if (SystemUtils.IS_OS_WINDOWS) {
      Some(resourcePath("win", lib))
    } else {
      // TODO: the only available .so file is compatible with TF 2.4.x - Current TF Java supports TF 2.3.1
      val libWithExtension = lib + ".so"
      Some(resourcePath("linux", libWithExtension))
    }

  private def getFileName(path: String) = {
    "sparknlp_sp_lib"+new File(path).getName.take(5)
  }

  /** NOT thread safe. Make sure this runs on DRIVER only*/
  private def copyResourceToTmp(path: String): File = {
    val stream = ResourceHelper.getResourceStream(path)
    val tmpFolder = System.getProperty("java.io.tmpdir")
    val tmp = Paths.get(tmpFolder, getFileName(path)).toFile
    val target = new BufferedOutputStream(new FileOutputStream(tmp))

    val buffer = new Array[Byte](1 << 13)
    var read = stream.read(buffer)
    while (read > 0) {
      target.write(buffer, 0, read)
      read = stream.read(buffer)
    }
    stream.close()
    target.close()

    tmp
  }

  def loadSPToCluster(spark: SparkSession): Unit = {
    /** NOT thread-safe. DRIVER only*/
    require(!SystemUtils.IS_OS_WINDOWS, "UniversalSentenceEncoder multi-lingual models are not supported on Windows.")

    if (!LoadSentencepiece.loadedToCluster && sentencepiecePaths.isDefined) {
      LoadSentencepiece.loadedToCluster = true
      spark.sparkContext.addFile(copyResourceToTmp(sentencepiecePaths.get).getPath)
    }
  }

  def loadSPToTensorflow(): Unit = {
    require(!SystemUtils.IS_OS_WINDOWS, "UniversalSentenceEncoder multi-lingual models are not supported on Windows.")

    if (!LoadSentencepiece.loadedToTensorflow && sentencepiecePaths.isDefined) {
      LoadSentencepiece.loadedToTensorflow = true
      val fp = SparkFiles.get(getFileName(sentencepiecePaths.get))
      if (new File(fp).exists()) {
        TensorFlow.loadLibrary(fp)
      }
    }
  }

  def loadSPToTensorflowLocally(): Unit = {
    require(!SystemUtils.IS_OS_WINDOWS, "UniversalSentenceEncoder multi-lingual models are not supported on Windows.")

    if (!LoadSentencepiece.loadedToTensorflow && sentencepiecePaths.isDefined) {
      LoadSentencepiece.loadedToTensorflow = true

      val path = sentencepiecePaths.get
      val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
      val inputStream = ResourceHelper.getResourceStream(uri.toString)

      // 1. Create tmp folder
      val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_sentencepiece")
        .toAbsolutePath.toString

      val spProcFile = new File(tmpFolder, getFileName(sentencepiecePaths.get))

      Files.copy(inputStream, spProcFile.toPath)
      val fp = spProcFile.toString

      if (new File(fp).exists()) {
        TensorFlow.loadLibrary(fp)
      }
    }
  }

}
