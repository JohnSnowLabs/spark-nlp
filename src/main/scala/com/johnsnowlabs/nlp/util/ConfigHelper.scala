package com.johnsnowlabs.nlp.util

import java.io.File

import com.typesafe.config.{Config, ConfigFactory}

object ConfigHelper {

  private val defaultConfig = ConfigFactory.load()

  def retrieve: Config = ConfigFactory
    .parseFile(new File(defaultConfig.getString("settings.overrideConfPath")))
    .withFallback(defaultConfig)

  lazy val config = retrieve

  def getConfigValue(path: String): Option[String] = {
    if (!config.hasPath(path))
      None
    else
      Some(config.getString(path))
  }

  def getConfigValueOrElse(path: String, defaultValue: => String): String = {
    getConfigValue(path).getOrElse(defaultValue)
  }

  // Configures s3 bucket where pretrained models are stored
  val pretrainedS3BucketKey = "nlp.pretrained.s3_bucket"

  // Configures s3 path where pretrained models are stored
  val pretrainedS3PathKey = "nlp.pretrained.s3_path"

  // Configures cache folder where to cache pretrained models
  val pretrainedCacheFolder = "nlp.pretrained.cache_folder"
}
