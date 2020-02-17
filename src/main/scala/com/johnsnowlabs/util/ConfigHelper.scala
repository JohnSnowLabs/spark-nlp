package com.johnsnowlabs.util

import com.johnsnowlabs.util.ConfigLoader.retrieve


object ConfigHelper {

  def hasPath(path: String): Boolean = {
    retrieve.hasPath(path)
  }

  def getConfigValue(path: String): Option[String] = {
    if (!retrieve.hasPath(path))
      None
    else
      Some(retrieve.getString(path))
  }

  def getConfigValueOrElse(path: String, defaultValue: => String): String = {
    getConfigValue(path).getOrElse(defaultValue)
  }

  val configPrefix = "jsl.sparknlp"

  // Configures s3 bucket where pretrained models are stored
  val pretrainedS3BucketKey = configPrefix + ".settings.pretrained.s3_bucket"

  // Configures s3 path where pretrained models are stored
  val pretrainedS3PathKey = configPrefix + ".settings.pretrained.s3_path"

  // Configures cache folder where to cache pretrained models
  val pretrainedCacheFolder = configPrefix + ".settings.pretrained.cache_folder"

  // Configures cache folder where to cache pretrained models
  val annotatorLogFolder = configPrefix + "settings.annotator.log_folder"

  // Stores credentials for AWS S3 private models
  val awsCredentials = configPrefix + ".settings.pretrained.credentials"


  val accessKeyId = awsCredentials + ".access_key_id"
  val secretAccessKey = awsCredentials + ".secret_access_key"
  val awsProfileName = awsCredentials + ".aws_profile_name"

  val s3SocketTimeout = configPrefix + ".settings.pretrained.s3_socket_timeout"

  val storageTmpDir = configPrefix + ".settings.storage.cluster_tmp_dir"

}
