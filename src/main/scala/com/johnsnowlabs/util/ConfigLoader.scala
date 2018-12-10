package com.johnsnowlabs.util

import java.io.File

import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.typesafe.config.{Config, ConfigFactory}

object ConfigLoader {

  private var defaultConfig = ConfigFactory.load()
  private var overrideConfigPath = defaultConfig.getString("sparknlp.settings.overrideConfigPath")

  def setConfigPath(path: String): Unit = {
    overrideConfigPath = path
    ResourceDownloader.resetResourceDownloader()
  }

  def getConfigPath: String = overrideConfigPath

  def retrieve: Config =
    ConfigFactory
      .parseFile(new File(overrideConfigPath))
      .withFallback(defaultConfig)

}
