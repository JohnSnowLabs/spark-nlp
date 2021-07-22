package com.johnsnowlabs.util

import java.io.File

import com.typesafe.config.{Config, ConfigFactory}


object ConfigLoader {


  private val defaultConfig = ConfigFactory.load()
  private val overrideConfigPath: String = {
    var configFile: String = ""
    if (System.getenv().containsKey("SPARKNLP_CONFIG_PATH")) {
      println("********** Getting config file from environment variable")
      configFile = System.getenv("SPARKNLP_CONFIG_PATH")
    } else {
      println("********** Getting config file from default file")
      configFile = defaultConfig.getString("sparknlp.settings.overrideConfigPath")
    }
    println(s"***************** ConfigFile=$configFile")
    configFile
  }

  def getConfigPath: String = overrideConfigPath

  def retrieve: Config = {

    ConfigFactory
      .parseFile(new File(overrideConfigPath))
      //.withFallback(defaultConfig) TODO: Uncomment after testing it
  }

}
