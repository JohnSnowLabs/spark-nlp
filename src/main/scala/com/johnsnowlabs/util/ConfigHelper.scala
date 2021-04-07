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

  // Configures s3 bucket where community pretrained models are stored
  val pretrainedCommunityS3BucketKey = "sparknlp.settings.pretrained.s3_community_bucket"

  // Configures s3 path where pretrained models are stored
  val pretrainedS3PathKey = "sparknlp.settings.pretrained.s3_path"

  // Configures cache folder where to cache pretrained models
  val pretrainedCacheFolder = "sparknlp.settings.pretrained.cache_folder"

  // Configures log folder where to store annotator logs using OutputHelper
  val annotatorLogFolder = "sparknlp.settings.annotator.log_folder"

  // Stores credentials for AWS S3 private models
  val awsCredentials = "sparknlp.settings.pretrained.credentials"


  val accessKeyId: String = awsCredentials + ".access_key_id"
  val secretAccessKey: String = awsCredentials + ".secret_access_key"
  val awsProfileName: String = awsCredentials + ".aws_profile_name"

  val s3SocketTimeout = "sparknlp.settings.pretrained.s3_socket_timeout"

  val storageTmpDir = "sparknlp.settings.storage.cluster_tmp_dir"

  val serializationMode: String = getConfigValueOrElse("sparknlp.settings.annotatorSerializationFormat", "object")
  val useBroadcast: Boolean = getConfigValueOrElse("sparknlp.settings.useBroadcastForFeatures", "true").toBoolean


}
