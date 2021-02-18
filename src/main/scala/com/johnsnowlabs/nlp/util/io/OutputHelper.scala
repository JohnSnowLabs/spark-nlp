package com.johnsnowlabs.nlp.util.io

import java.io.{File, FileWriter, PrintWriter}

import com.johnsnowlabs.util.ConfigHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import scala.language.existentials


object OutputHelper {

  lazy private val fs = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)

  lazy private val homeDirectory = if (fs.getScheme.equals("dbfs")) System.getProperty("user.home") else fs.getHomeDirectory

  private def logsFolder: String = ConfigHelper.getConfigValueOrElse(ConfigHelper.annotatorLogFolder, homeDirectory + "/annotator_logs")


  lazy private val isDBFS = fs.getScheme.equals("dbfs")

  def writeAppend(uuid: String, content: String, outputLogsPath: String): Unit = {

    val targetFolder = if (outputLogsPath.isEmpty) logsFolder else outputLogsPath

    if (isDBFS) {
      if (!new File(targetFolder).exists()) new File(targetFolder).mkdirs()
    }else{
      if (!fs.exists(new Path(targetFolder))) fs.mkdirs(new Path(targetFolder))
    }

    val targetPath = new Path(targetFolder, uuid + ".log")

    if (fs.getScheme.equals("file") || fs.getScheme.equals("dbfs")) {
      val fo = new File(targetPath.toUri.getRawPath)
      val writer = new FileWriter(fo, true)
      writer.append(content + System.lineSeparator())
      writer.close()
    } else {
      fs.createNewFile(targetPath)
      val fo = fs.append(targetPath)
      val writer = new PrintWriter(fo, true)
      writer.append(content + System.lineSeparator())
      writer.close()
      fo.close()

    }
  }

}
