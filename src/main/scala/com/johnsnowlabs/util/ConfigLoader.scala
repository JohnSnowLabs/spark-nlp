package com.johnsnowlabs.util

import java.io.File

import com.typesafe.config.{Config, ConfigFactory}

object ConfigLoader {

  private val defaultConfig = ConfigFactory.load()
  private var overrideConfigPath = defaultConfig.getString("sparknlp.settings.overrideConfigPath")

  def setConfigPath(path: String): Unit = overrideConfigPath = path

  def getConfigPath: String = overrideConfigPath

  def retrieve: Config = ConfigFactory
    .parseFile(new File(overrideConfigPath))
    .withFallback(defaultConfig)
}
