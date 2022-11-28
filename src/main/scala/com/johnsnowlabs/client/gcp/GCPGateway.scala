package com.johnsnowlabs.client.gcp

import com.google.cloud.storage.{BlobId, BlobInfo, Storage, StorageOptions}
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}

import java.io.InputStream

class GCPGateway(
    projectId: String = ConfigLoader.getConfigStringValue(ConfigHelper.gcpProjectId)) {

  lazy val storageClient: Storage = {
    if (projectId == null || projectId.isEmpty) {
      throw new UnsupportedOperationException(
        "projectId argument is mandatory to create GCP Storage client.")
    }

    StorageOptions.newBuilder.setProjectId(projectId).build.getService
  }

  def copyFileToGCPStorage(
      bucket: String,
      destinationGCPStoragePath: String,
      inputStream: InputStream): Unit = {

    val blobId = BlobId.of(bucket, destinationGCPStoragePath)
    val blobInfo = BlobInfo.newBuilder(blobId).build

    storageClient.createFrom(blobInfo, inputStream)
  }

}
