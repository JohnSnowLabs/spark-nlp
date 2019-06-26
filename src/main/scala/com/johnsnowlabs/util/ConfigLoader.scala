package com.johnsnowlabs.util

import java.io.File

import com.typesafe.config.{Config, ConfigFactory}


object ConfigLoader {


  private val defaultConfig = ConfigFactory.load()
  private val overrideConfigPath: String = {
    if (System.getenv().containsKey("SPARKNLP_CONFIG_PATH"))
      System.getenv("SPARKNLP_CONFIG_PATH")
    else
      defaultConfig.getString("sparknlp.settings.overrideConfigPath")
  }

  def getConfigPath: String = overrideConfigPath

  def retrieve: Config = {

    ConfigFactory
      .parseFile(new File(overrideConfigPath))
      .withFallback(defaultConfig)
  }

}
