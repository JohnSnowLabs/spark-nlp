package com.johnsnowlabs.nlp.util

import java.io.File

import com.typesafe.config.{Config, ConfigFactory}

object ConfigHelper {

  private val defaultConfig = ConfigFactory.load()
  def retrieve: Config = ConfigFactory
    .parseFile(new File(defaultConfig.getString("settings.overrideConfPath")))
    .withFallback(defaultConfig)

}
