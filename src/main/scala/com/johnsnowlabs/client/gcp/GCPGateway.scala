package com.johnsnowlabs.client.gcp

import com.google.cloud.storage.{BlobId, BlobInfo, Storage, StorageOptions}
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.sparkproject.guava.collect.Iterables

import java.io.InputStream

class GCPGateway(
    projectId: String = ConfigLoader.getConfigStringValue(ConfigHelper.gcpProjectId)) {

  private lazy val storageClient: Storage = {
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

  def doesFolderExist(bucket: String, prefix: String): Boolean = {
    val storage = StorageOptions.newBuilder.setProjectId(projectId).build.getService
    val blobs = storage.list(
      bucket,
      Storage.BlobListOption.prefix(prefix),
      Storage.BlobListOption.currentDirectory)

    val blobsSize = Iterables.size(blobs.iterateAll())
    blobsSize > 0
  }

}
