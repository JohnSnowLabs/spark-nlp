package com.johnsnowlabs.client.azure

import com.johnsnowlabs.client.util.CloudHelper
import com.johnsnowlabs.client.{CloudClient, CloudStorage}
import com.johnsnowlabs.util.ConfigHelper

class AzureClient(parameters: Map[String, String] = Map.empty) extends CloudClient {

  private lazy val azureStorageConnection = cloudConnect()

  override protected def cloudConnect(): CloudStorage = {
    if (CloudHelper.isMicrosoftFabric) {
      // These params are NOT required for Fabric
      new AzureGateway("", "", isFabricLakehouse = true)
    } else {
      val storageAccountName = parameters.getOrElse(
        "storageAccountName",
        throw new Exception("Azure client requires storageAccountName"))
      val accountKey =
        parameters.getOrElse("accountKey", ConfigHelper.getHadoopAzureConfig(storageAccountName))
      new AzureGateway(storageAccountName, accountKey)
    }
  }

  override def doesBucketPathExist(bucketName: String, filePath: String): Boolean = {
    azureStorageConnection.doesBucketPathExist(bucketName, filePath)
  }

  override def copyInputStreamToBucket(
      bucketName: String,
      filePath: String,
      sourceFilePath: String): Unit = {
    azureStorageConnection.copyInputStreamToBucket(bucketName, filePath, sourceFilePath)
  }

  override def downloadFilesFromBucketToDirectory(
      bucketName: String,
      filePath: String,
      directoryPath: String,
      isIndex: Boolean): Unit = {
    azureStorageConnection.downloadFilesFromBucketToDirectory(
      bucketName,
      filePath,
      directoryPath,
      isIndex)
  }

}
