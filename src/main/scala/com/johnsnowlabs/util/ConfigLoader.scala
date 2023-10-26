/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.util.io.OutputHelper
import org.apache.hadoop.fs.FileSystem

import java.util.UUID
import scala.util.{Failure, Success, Try}

object ConfigLoader {

  private lazy val fileSystem: FileSystem = OutputHelper.getFileSystem
  private lazy val homeDirectory: String = {
    if (fileSystem.getScheme.equals("dbfs")) System.getProperty("user.home")
    else fileSystem.getHomeDirectory.toString
  }
  private lazy val hadoopTmpDir: String = ConfigHelper.getHadoopTmpDir

  private lazy val configData: Map[String, String] = {

    getConfigInfo(ConfigHelper.pretrainedS3BucketKey, "auxdata.johnsnowlabs.com") ++
      getConfigInfo(ConfigHelper.pretrainedCommunityS3BucketKey, "community.johnsnowlabs.com") ++
      getConfigInfo(ConfigHelper.pretrainedS3PathKey, "") ++
      getConfigInfo(ConfigHelper.pretrainedCacheFolder, homeDirectory + "/cache_pretrained") ++
      getConfigInfo(ConfigHelper.annotatorLogFolder, homeDirectory + "/annotator_logs") ++
      getConfigInfo(ConfigHelper.accessKeyId, "") ++
      getConfigInfo(ConfigHelper.secretAccessKey, "") ++
      getConfigInfo(ConfigHelper.sessionToken, "") ++
      getConfigInfo(ConfigHelper.awsProfileName, "") ++
      getConfigInfo(ConfigHelper.awsRegion, "") ++
      getConfigInfo(ConfigHelper.s3SocketTimeout, "0") ++
      getConfigInfo(ConfigHelper.storageTmpDir, hadoopTmpDir) ++
      getConfigInfo(ConfigHelper.serializationMode, "object") ++
      getConfigInfo(ConfigHelper.useBroadcast, "true") ++
      getConfigInfo(ConfigHelper.awsExternalAccessKeyId, "") ++
      getConfigInfo(ConfigHelper.awsExternalSecretAccessKey, "") ++
      getConfigInfo(ConfigHelper.awsExternalSessionToken, "") ++
      getConfigInfo(ConfigHelper.awsExternalProfileName, "") ++
      getConfigInfo(ConfigHelper.awsExternalS3BucketKey, "") ++
      getConfigInfo(ConfigHelper.awsExternalRegion, "") ++
      getConfigInfo(ConfigHelper.gcpProjectId, "") ++
      getConfigInfo(ConfigHelper.openAIAPIKey, sys.env.getOrElse("OPENAI_API_KEY", "")) ++
      getConfigInfo(ConfigHelper.onnxGpuDeviceId, "0") ++
      getConfigInfo(ConfigHelper.onnxIntraOpNumThreads, "6") ++
      getConfigInfo(ConfigHelper.onnxOptimizationLevel, "ALL_OPT") ++
      getConfigInfo(ConfigHelper.onnxExecutionMode, "SEQUENTIAL")
  }

  private def getConfigInfo(property: String, defaultValue: String): Map[String, String] = {
    if (property == ConfigHelper.storageTmpDir) {
      val path = ConfigHelper.getConfigValueOrElse(property, defaultValue)
      val tmpLocation = path + "/" + UUID.randomUUID().toString.takeRight(12) + "_cdx"
      Map(property -> tmpLocation)
    } else {
      Map(property -> ConfigHelper.getConfigValueOrElse(property, defaultValue))
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

}
