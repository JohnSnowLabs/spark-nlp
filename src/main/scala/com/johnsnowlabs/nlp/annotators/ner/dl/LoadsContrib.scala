package com.johnsnowlabs.nlp.annotators.ner.dl

import java.io.{BufferedOutputStream, File, FileOutputStream}
import java.nio.file.Paths

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.commons.lang.SystemUtils
import org.apache.spark.SparkFiles
import org.apache.spark.sql.SparkSession
import org.tensorflow.TensorFlow

trait LoadsContrib {

  private lazy val lib1 = "_sparse_feature_cross_op.so"
  private lazy val lib2 = "_lstm_ops.so"

  private def resourcePath(os: String, lib: String) = "ner-dl/"+os+"/"+lib

  lazy val contribPaths: Option[(String, String)] =
    if (SystemUtils.IS_OS_MAC || SystemUtils.IS_OS_MAC_OSX) {
      Some((resourcePath("mac",lib1), resourcePath("mac", lib2)))
    } else if (SystemUtils.IS_OS_WINDOWS) {
      None
    } else {
      Some((resourcePath("linux",lib1), resourcePath("linux", lib2)))
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
      println(s"adding ${contribPaths.get}")
      LoadsContrib.loadedToCluster = true
      spark.sparkContext.addFile(copyResourceToTmp(contribPaths.get._1).getPath)
      spark.sparkContext.addFile(copyResourceToTmp(contribPaths.get._2).getPath)
    }
  }

  def loadContribToTensorflow(): Unit = {
    if (!LoadsContrib.loadedToTensorflow && contribPaths.isDefined) {
      println("loading to tensorflow")
      LoadsContrib.loadedToTensorflow = true
      TensorFlow.loadLibrary(SparkFiles.get(getFileName(contribPaths.get._1)))
      TensorFlow.loadLibrary(SparkFiles.get(getFileName(contribPaths.get._2)))
    }
  }

}
object LoadsContrib {
  @transient var loadedToCluster = false
  @transient var loadedToTensorflow = false
}