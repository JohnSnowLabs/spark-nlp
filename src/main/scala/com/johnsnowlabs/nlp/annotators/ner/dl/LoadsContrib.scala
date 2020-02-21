package com.johnsnowlabs.nlp.annotators.ner.dl

import java.io.{BufferedOutputStream, File, FileOutputStream}
import java.nio.file.Paths

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.commons.lang.SystemUtils
import org.apache.spark.SparkFiles
import org.apache.spark.sql.SparkSession
import org.tensorflow.TensorFlow

object LoadsContrib {
  @transient var loadedToCluster = false
  @transient var loadedToTensorflow = false

  private lazy val lib1 = "_sparse_feature_cross_op.so"
  private lazy val lib2 = "_lstm_ops.so"

  private def resourcePath(os: String, lib: String) = "ner-dl/"+os+"/"+lib

  /*
  * In TensorFlow 1.15.0 we don't need to load any .so files
  * We reserve this feature for the future releases
  *  */
  lazy val contribPaths: Option[(String, String)] =
    if (SystemUtils.IS_OS_MAC || SystemUtils.IS_OS_MAC_OSX) {
      None
    } else if (SystemUtils.IS_OS_WINDOWS) {
      None
    } else {
      None
    }

  private def getFileName(path: String) = {
    "sparknlp_contrib"+new File(path).getName.take(5)
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

  def loadContribToCluster(spark: SparkSession): Unit = {
    /** NOT thread-safe. DRIVER only*/
    if (!LoadsContrib.loadedToCluster && contribPaths.isDefined) {
      LoadsContrib.loadedToCluster = true
      spark.sparkContext.addFile(copyResourceToTmp(contribPaths.get._1).getPath)
      spark.sparkContext.addFile(copyResourceToTmp(contribPaths.get._2).getPath)
    }
  }

  def loadContribToTensorflow(): Unit = {
    if (!LoadsContrib.loadedToTensorflow && contribPaths.isDefined) {
      LoadsContrib.loadedToTensorflow = true
      val fp1 = SparkFiles.get(getFileName(contribPaths.get._1))
      val fp2 = SparkFiles.get(getFileName(contribPaths.get._2))
      if (new File(fp1).exists() && new File(fp2).exists()) {
        TensorFlow.loadLibrary(fp1)
        TensorFlow.loadLibrary(fp2)
      }
    }
  }

}
