package com.johnsnowlabs.nlp.util.io

import com.johnsnowlabs.util.{ConfigHelperV2, ConfigLoaderV2}
import org.apache.hadoop.fs.{FileSystem, Path}

import java.io.{File, FileWriter, PrintWriter}
import scala.language.existentials


object OutputHelper {

  lazy private val fileSystem = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)

//  lazy private val homeDirectory = if (fileSystem.getScheme.equals("dbfs")) System.getProperty("user.home") else fileSystem.getHomeDirectory
//
//  private def logsFolder: String = {
//    val path = ConfigHelper.annotatorLogFolder
//    val defaultValue = homeDirectory + "/annotator_logs"
//    ConfigHelper.getConfigValueOrElse(path, defaultValue)
//  }

  private def logsFolder: String = ConfigLoaderV2.getConfigStringValue(ConfigHelperV2.annotatorLogFolder)

  lazy private val isDBFS = fileSystem.getScheme.equals("dbfs")

  def writeAppend(uuid: String, content: String, outputLogsPath: String): Unit = {
    println(s"**************** In OutputHelper.writeAppend outputLogsPath=$outputLogsPath")
    val targetFolder = if (outputLogsPath.isEmpty) logsFolder else outputLogsPath

    if (isDBFS) {
      if (!new File(targetFolder).exists()) new File(targetFolder).mkdirs()
    }else{
      if (!fileSystem.exists(new Path(targetFolder))) fileSystem.mkdirs(new Path(targetFolder))
    }

    val targetPath = new Path(targetFolder, uuid + ".log")

    if (fileSystem.getScheme.equals("file") || fileSystem.getScheme.equals("dbfs")) {
      val fo = new File(targetPath.toUri.getRawPath)
      val writer = new FileWriter(fo, true)
      writer.append(content + System.lineSeparator())
      writer.close()
    } else {
      fileSystem.createNewFile(targetPath)
      val fo = fileSystem.append(targetPath)
      val writer = new PrintWriter(fo, true)
      writer.append(content + System.lineSeparator())
      writer.close()
      fo.close()

    }
  }

}
