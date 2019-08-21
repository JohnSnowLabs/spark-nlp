package com.johnsnowlabs.nlp.util.io

import com.johnsnowlabs.util.ConfigHelper
import org.apache.hadoop.fs.{FileSystem, Path}

object OutputHelper {

  lazy private val fs = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)

  private def cacheFolder: String = ConfigHelper.getConfigValueOrElse(ConfigHelper.annotatorLogFolder, fs.getHomeDirectory + "/annotator_logs")

  def writeAppend(uuid: String, content: String): Unit = {
    val target = fs.append(new Path(cacheFolder, uuid+".log"))
    target.writeChars(content)
    target.close()
  }

}
