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

  // Configures s3 bucket where pretrained models are stored
  val pretrainedS3BucketKey = "sparknlp.settings.pretrained.s3_bucket"

  // Configures s3 path where pretrained models are stored
  val pretrainedS3PathKey = "sparknlp.settings.pretrained.s3_path"

  // Configures cache folder where to cache pretrained models
  val pretrainedCacheFolder = "sparknlp.settings.pretrained.cache_folder"

  // Configures cache folder where to cache pretrained models
  val annotatorLogFolder = "sparknlp.settings.annotator.log_folder"

  // Stores credentials for AWS S3 private models
  val awsCredentials = "sparknlp.settings.pretrained.credentials"


  val accessKeyId = awsCredentials + ".access_key_id"
  val secretAccessKey = awsCredentials + ".secret_access_key"
  val awsProfileName = awsCredentials + ".aws_profile_name"

  val s3SocketTimeout = "sparknlp.settings.pretrained.s3_socket_timeout"

  val storageTmpDir = "sparknlp.settings.storage.cluster_tmp_dir"

}
