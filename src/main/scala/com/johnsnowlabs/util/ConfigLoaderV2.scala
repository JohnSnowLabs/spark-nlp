package com.johnsnowlabs.util

import org.apache.hadoop.fs.FileSystem

import java.util.UUID
import scala.util.{Failure, Success, Try}

object ConfigLoaderV2 {

  private lazy val fileSystem: FileSystem = ConfigHelperV2.getFileSystem
  private lazy val homeDirectory = {
    if (fileSystem.getScheme.equals("dbfs")) System.getProperty("user.home") else fileSystem.getHomeDirectory
  }
  private lazy val hadoopTmpDir: String = ConfigHelperV2.getHadoopTmpDir

  private lazy val configData: Map[String, String] = {

    getConfigInfo(ConfigHelperV2.pretrainedS3BucketKey, "auxdata.johnsnowlabs.com") ++
    getConfigInfo(ConfigHelperV2.pretrainedCommunityS3BucketKey, "community.johnsnowlabs.com") ++
    getConfigInfo(ConfigHelperV2.pretrainedS3PathKey, "") ++
    getConfigInfo(ConfigHelperV2.pretrainedCacheFolder, fileSystem.getHomeDirectory + "/cache_pretrained") ++ //TODO: Check using homeDirectory as in logFolder
    getConfigInfo(ConfigHelperV2.annotatorLogFolder, homeDirectory + "/annotator_logs") ++
    getConfigInfo(ConfigHelperV2.accessKeyId, "") ++
    getConfigInfo(ConfigHelperV2.secretAccessKey, "") ++
    getConfigInfo(ConfigHelperV2.awsProfileName, "") ++
    getConfigInfo(ConfigHelperV2.s3SocketTimeout, "0") ++
    getConfigInfo(ConfigHelperV2.storageTmpDir, hadoopTmpDir) ++
    getConfigInfo(ConfigHelperV2.serializationMode, "object") ++
    getConfigInfo(ConfigHelperV2.useBroadcast, "true")

  }

  private def getConfigInfo(property: String, defaultValue: String): Map[String, String] = {
    if (property== ConfigHelperV2.storageTmpDir) {
      val path = ConfigHelperV2.getConfigValueOrElse(property, defaultValue)
      val tmpLocation = path + "/" + UUID.randomUUID().toString.takeRight(12) + "_cdx"
      Map(property -> tmpLocation)
    } else {
      Map(property -> ConfigHelperV2.getConfigValueOrElse(property, defaultValue))
    }
  }

  def getConfigStringValue(property: String): String = {
    configData.getOrElse(property, "")
  }

  def getConfigIntValue(property: String): Int = {
    val value: String = configData.getOrElse(property, "0")
    toInt(value) match {
      case Success(value) => value
      case Failure(_) => 0
    }
  }

  private def toInt(string: String): Try[Int] = Try {
    Integer.parseInt(string.trim)
  }

  def getConfigBooleanValue(property: String): Boolean = {
    val value: String = configData.getOrElse(property, "true")
    toBoolean(value) match {
      case Success(value) => value
      case Failure(_) => true
    }
  }

  private def toBoolean(string: String): Try[Boolean] = Try {
    java.lang.Boolean.parseBoolean(string.trim)
  }

  def hasAwsCredentials: Boolean = {
    val hasAccessKeyId = getConfigStringValue(ConfigHelperV2.accessKeyId) != ""
    val hasSecretAccessKey = getConfigStringValue(ConfigHelperV2.secretAccessKey) != ""
    val hasAwsProfileName = getConfigStringValue(ConfigHelperV2.awsProfileName) != ""
    if (hasAwsProfileName || hasAccessKeyId || hasSecretAccessKey) true else false
  }

}
