package com.johnsnowlabs.nlp.util.io

import java.io.{File, FileWriter, PrintWriter}

import com.johnsnowlabs.util.ConfigHelper
import org.apache.hadoop.fs.{FileSystem, Path}

object OutputHelper {

  lazy private val fs = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)

  private def cacheFolder: String = ConfigHelper.getConfigValueOrElse(ConfigHelper.annotatorLogFolder, fs.getHomeDirectory + "/annotator_logs")

  def writeAppend(uuid: String, content: String): Unit = {
    val targetPath = new Path(cacheFolder, uuid+".log")
    if (fs.getScheme != "file") {
      val fo = fs.append(targetPath)
      val writer = new PrintWriter(fo, true)
      writer.append(content + System.lineSeparator())
      writer.close()
      fo.close()
    } else {
      val fo = new File(targetPath.toUri.getRawPath)
      val writer = new FileWriter(fo, true)
      writer.append(content + System.lineSeparator())
      writer.close()
    }
  }

}
