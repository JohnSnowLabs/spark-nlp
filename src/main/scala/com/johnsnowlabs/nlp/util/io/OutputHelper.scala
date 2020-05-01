package com.johnsnowlabs.nlp.util.io

import java.io.{File, FileWriter, PrintWriter}

import com.johnsnowlabs.util.ConfigHelper
import org.apache.hadoop.fs.{FileSystem, Path}

object OutputHelper {

  lazy private val fs = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)

  private def logsFolder: String = ConfigHelper.getConfigValueOrElse(ConfigHelper.annotatorLogFolder, fs.getHomeDirectory + "/annotator_logs")

  private lazy val logsFolderExists = fs.exists(new Path(logsFolder))

  def writeAppend(uuid: String, content: String, outputLogsPath: String): Unit = {

    val targetPath = if(outputLogsPath.isEmpty){
      if (!logsFolderExists) fs.mkdirs(new Path(logsFolder))
       new Path(logsFolder, uuid+".log")
    }else{
      if (!fs.exists(new Path(outputLogsPath))) fs.mkdirs(new Path(outputLogsPath))
      new Path(outputLogsPath, uuid+".log")
    }

    if (fs.getScheme.equals("file") || fs.getScheme.equals("dbfs")) {
      val fo = new File(targetPath.toUri.getRawPath)
      val writer = new FileWriter(fo, true)
      writer.append(content + System.lineSeparator())
      writer.close()
    } else {
      val fo = fs.append(targetPath)
      val writer = new PrintWriter(fo, true)
      writer.append(content + System.lineSeparator())
      writer.close()
      fo.close()

    }
  }

}
