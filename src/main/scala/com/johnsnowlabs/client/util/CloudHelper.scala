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
package com.johnsnowlabs.client.util

import com.johnsnowlabs.nlp.util.io.CloudStorageType.CloudStorageType
import com.johnsnowlabs.nlp.util.io.{CloudStorageType, ResourceHelper}

import java.net.{URI, URL}

object CloudHelper {

  def parseS3URI(s3URI: String, includePrefixInKey: Boolean = false): (String, String) = {
    val prefix = if (s3URI.startsWith("s3:")) "s3://" else "s3a://"
    val bucketName = s3URI.substring(prefix.length).split("/").head
    val key = s3URI.substring((prefix + bucketName).length + 1)

    require(bucketName.nonEmpty, "S3 bucket name is empty!")
    (bucketName, if (includePrefixInKey) prefix + key else key)
  }

  def parseGCPStorageURI(gcpStorageURI: String): (String, String) = {
    val prefix = "gs://"
    val bucketName = gcpStorageURI.substring(prefix.length).split("/").head
    val storagePath = gcpStorageURI.substring((prefix + bucketName).length + 1)

    require(bucketName.nonEmpty, "GCP Storage bucket name is empty!")

    (bucketName, storagePath)
  }

  def parseAzureBlobURI(azureBlobURI: String): (String, String) = {
    val uri = new URI(azureBlobURI)
    val parts = uri.getPath.stripPrefix("/").split("/", 2)
    val containerName = parts(0)
    require(containerName.nonEmpty, "Azure container name is empty!")
    val blobPath = if (parts.length > 1) parts(1) else ""

    (containerName, blobPath)
  }

  def getAccountNameFromAzureBlobURI(azureBlobURI: String): String = {
    val uri = new URI(azureBlobURI)
    val host = uri.getHost
    val accountName = host.stripSuffix(".blob.core.windows.net")
    require(accountName.nonEmpty, "Azure storage account name is empty!")
    accountName
  }

  def transformURIToWASB(azureURI: String): String = {
    val url = new URL(azureURI)
    val host = url.getHost
    val pathParts = url.getPath.split("/").filter(_.nonEmpty)
    val container = if (pathParts.nonEmpty) pathParts(0) else ""
    require(container.nonEmpty, "Azure container name is empty!")
    val pathWithoutContainer = if (pathParts.length > 1) pathParts.drop(1).mkString("/") else ""

    s"wasbs://$container@$host/$pathWithoutContainer/"
  }

  def isCloudPath(uri: String): Boolean = {
    isS3Path(uri) || isGCPStoragePath(uri) || isAzureBlobPath(uri)
  }

  def isS3Path(uri: String): Boolean = {
    uri.startsWith("s3://") || uri.startsWith("s3a://")
  }

  private def isGCPStoragePath(uri: String): Boolean = uri.startsWith("gs://")

  private def isAzureBlobPath(uri: String): Boolean = {
    (uri.startsWith("https://") && uri.contains(".blob.core.windows.net/")) || uri.startsWith(
      "abfss://")
  }

  def isMicrosoftFabric: Boolean = {
    ResourceHelper.spark.conf.getAll.keys.exists(_.startsWith("spark.fabric"))
  }

  def isFabricAbfss(uri: String): Boolean =
    uri.startsWith("abfss://") && uri.contains("onelake.dfs.fabric.microsoft.com")

  def cloudType(uri: String): CloudStorageType = {
    if (isS3Path(uri)) {
      CloudStorageType.S3
    } else if (isGCPStoragePath(uri)) {
      CloudStorageType.GCP
    } else if (isAzureBlobPath(uri)) {
      CloudStorageType.Azure
    } else throw new UnsupportedOperationException(s"Unsupported URI scheme: $uri")
  }

}
