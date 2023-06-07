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
package com.johnsnowlabs.client

import com.johnsnowlabs.client.aws.AWSClient
import com.johnsnowlabs.client.gcp.GCPClient
import com.johnsnowlabs.client.util.CloudHelper
import com.johnsnowlabs.nlp.util.io.CloudStorageType
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.slf4j.{Logger, LoggerFactory}

class CloudManager(parameters: Map[String, String] = Map.empty) {

  private val logger: Logger = LoggerFactory.getLogger(this.getClass.toString)

  def getClientInstance(uri: String): CloudClient = {
    CloudHelper.cloudType(uri) match {
      case CloudStorageType.S3 =>
        new AWSClient(parameters)
      case CloudStorageType.GCP => {
        new GCPClient(parameters)
      }
      case _ =>
        throw new IllegalArgumentException(s"Unsupported URI scheme: $uri")
    }
  }

  def getClientInstanceFromConfigurationParams(uri: String): CloudClient = {
    CloudHelper.cloudType(uri) match {
      case CloudStorageType.S3 =>
        val (accessKey, secretKey, sessionToken) = ConfigHelper.getHadoopS3Config
        val region = ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalRegion)
        val isS3Defined =
          accessKey != null && secretKey != null && sessionToken != null && region.nonEmpty

        if (isS3Defined) {
          Map(
            "accessKeyId" -> accessKey,
            "secretAccessKey" -> secretKey,
            "sessionToken" -> sessionToken,
            "region" -> region)
        } else {
          if (accessKey != null || secretKey != null || sessionToken != null)
            logger.info(
              "Not all configs set for private S3 access. Defaulting to public downloader.")
          Map("credentialsType" -> "public")
        }
        new AWSClient(parameters)
      case CloudStorageType.GCP => {
        val projectId = ConfigLoader.getConfigStringValue(ConfigHelper.gcpProjectId)
        Map("projectId" -> projectId)
        new GCPClient(parameters)
      }
      //      case azureUri
      //          if azureUri.startsWith("https://") && azureUri.contains(".blob.core.windows.net/") => "Azure"
      case _ =>
        throw new IllegalArgumentException(s"Unsupported URI scheme: $uri")
    }
  }

}
