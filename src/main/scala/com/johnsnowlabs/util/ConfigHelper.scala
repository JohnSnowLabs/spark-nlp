package com.johnsnowlabs.util

import com.johnsnowlabs.util.ConfigLoader.retrieve


object ConfigHelper {

  def hasPath(path: String): Boolean = {
    retrieve.hasPath(path)
  }

  def getConfigValue[T](path: String): Option[T] = {
    if (!retrieve.hasPath(path))
      None
    else
      Some(retrieve.getAnyRef(path).asInstanceOf[T])
  }

  def getConfigValueOrElse(path: String, defaultValue: => String): String = {
    getConfigValue[String](path).getOrElse(defaultValue)
  }

  // Configures s3 bucket where pretrained models are stored
  val pretrainedS3BucketKey = "nlp.pretrained.s3_bucket"

  // Configures s3 path where pretrained models are stored
  val pretrainedS3PathKey = "nlp.pretrained.s3_path"

  // Configures cache folder where to cache pretrained models
  val pretrainedCacheFolder = "nlp.pretrained.cache_folder"

  // Stores credentials for AWS S3 private models
  val awsCredentials = "nlp.pretrained.credentials"

  val accessKeyId = awsCredentials + ".access_key_id"
  val secretAccessKey = awsCredentials + ".secret_access_key"

  val s3SocketTimeout = "nlp.pretrained.s3_path"

}
