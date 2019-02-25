package com.johnsnowlabs.util

import java.io.File

import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.hadoop.fs.{FileSystem, Path}


object ConfigLoader {


  private var defaultConfig = ConfigFactory.load()
  private var overrideConfigPath = defaultConfig.getString("sparknlp.settings.overrideConfigPath")


  def setConfigPath(path: String): Unit = {

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    if(uri.getScheme==null|| uri.getScheme.equalsIgnoreCase("file")){
      overrideConfigPath=path

    }else{
      val fs = FileSystem.get(uri, ResourceHelper.spark.sparkContext.hadoopConfiguration)
      val src= new Path(path)
      val dst= new Path(ResourceDownloader.cacheFolder,src.getName)
      fs.copyToLocalFile(src,dst)
      overrideConfigPath=dst.toUri.getPath
    }
    ResourceDownloader.resetResourceDownloader()
  }

  def getConfigPath: String = overrideConfigPath

  def retrieve: Config = {

    ConfigFactory
      .parseFile(new File(overrideConfigPath))
      .withFallback(defaultConfig)
  }

}
