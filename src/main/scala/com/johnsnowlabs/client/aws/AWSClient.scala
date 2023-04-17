/*
 * Copyright 2017-2023 John Snow Labs
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
package com.johnsnowlabs.client.aws

import com.johnsnowlabs.client.{CloudClient, CloudStorage}
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}

class AWSClient(parameters: Map[String, String] = Map.empty) extends CloudClient {

  private lazy val awsStorageConnection = cloudConnect()

  override protected def cloudConnect(): CloudStorage = {
    val accessKeyId = parameters.getOrElse(
      "accessKeyId",
      ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalAccessKeyId))
    val secretAccessKey = parameters.getOrElse(
      "secretAccessKey",
      ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalSecretAccessKey))
    val sessionToken = parameters.getOrElse(
      "sessionToken",
      ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalSessionToken))
    val awsProfile = parameters.getOrElse(
      "awsProfile",
      ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalProfileName))
    val region = parameters.getOrElse(
      "region",
      ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalRegion))
    val credentialsType = parameters.getOrElse("credentialsType", "private")

    if (parameters.isEmpty) {
      getAWSClientInstanceFromHadoopConfig
    } else {
      new AWSGateway(
        accessKeyId,
        secretAccessKey,
        sessionToken,
        awsProfile,
        region,
        credentialsType)
    }
  }

  private def getAWSClientInstanceFromHadoopConfig: AWSGateway = {
    var (accessKeyId, secretKey, sessionToken) = ConfigHelper.getHadoopS3Config
    if (accessKeyId == null) accessKeyId = ""
    if (secretKey == null) secretKey = ""
    if (sessionToken == null) sessionToken = ""

    if (accessKeyId == "" && secretKey == "") {
      throw new IllegalAccessException(
        "Empty access.key and secret.key hadoop configuration and parameters.")
    }
    val awsDestinationGateway = new AWSGateway(accessKeyId, secretKey, sessionToken)
    awsDestinationGateway
  }

  override def doesBucketPathExist(bucketName: String, filePath: String): Boolean = {
    awsStorageConnection.doesBucketPathExist(bucketName, filePath)
  }

  override def copyInputStreamToBucket(
      bucketName: String,
      filePath: String,
      sourceFilePath: String): Unit = {
    awsStorageConnection.copyInputStreamToBucket(bucketName, filePath, sourceFilePath)
  }

  override def downloadFilesFromBucketToDirectory(
      bucketName: String,
      filePath: String,
      directoryPath: String,
      isIndex: Boolean): Unit = {
    awsStorageConnection.downloadFilesFromBucketToDirectory(
      bucketName,
      filePath,
      directoryPath,
      isIndex)
  }

}
