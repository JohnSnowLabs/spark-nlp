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
package com.johnsnowlabs.client.gcp

import com.google.cloud.storage.{BlobId, BlobInfo, Storage, StorageOptions}
import com.johnsnowlabs.client.CloudStorage
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.sparkproject.guava.collect.Iterables

import java.io.{File, InputStream}
import scala.collection.JavaConverters._

class GCPGateway(projectId: String = ConfigLoader.getConfigStringValue(ConfigHelper.gcpProjectId))
    extends CloudStorage {

  private lazy val storageClient: Storage = {
    if (projectId == null || projectId.isEmpty) {
      throw new UnsupportedOperationException(
        "projectId argument is mandatory to create GCP Storage client.")
    }

    StorageOptions.newBuilder.setProjectId(projectId).build.getService
  }

  override def doesBucketPathExist(bucketName: String, filePath: String): Boolean = {
    val storage = StorageOptions.newBuilder.setProjectId(projectId).build.getService
    val blobs = storage.list(
      bucketName,
      Storage.BlobListOption.prefix(filePath),
      Storage.BlobListOption.currentDirectory)

    val blobsSize = Iterables.size(blobs.iterateAll())
    blobsSize > 0
  }

  override def copyFileToBucket(
      bucketName: String,
      destinationPath: String,
      inputStream: InputStream): Unit = {
    val blobId = BlobId.of(bucketName, destinationPath)
    val blobInfo = BlobInfo.newBuilder(blobId).build
    storageClient.createFrom(blobInfo, inputStream)
  }

  override def copyInputStreamToBucket(
      bucketName: String,
      filePath: String,
      sourceFilePath: String): Unit = {
    val fileSystem = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)
    val inputStream = fileSystem.open(new Path(sourceFilePath))
    val blobInfo = BlobInfo.newBuilder(bucketName, filePath).build()
    storageClient.createFrom(blobInfo, inputStream)
  }

  override def downloadFilesFromBucketToDirectory(
      bucketName: String,
      filePath: String,
      directoryPath: String,
      isIndex: Boolean = false): Unit = {
    try {
      val blobs = storageClient
        .list(bucketName, Storage.BlobListOption.prefix(filePath))
        .getValues
        .asScala
        .toArray

      blobs.foreach { blob =>
        val blobName = blob.getName
        val file = new File(s"$directoryPath/$blobName")

        if (blobName.endsWith("/")) {
          file.mkdirs()
        } else {
          file.getParentFile.mkdirs()
          blob.downloadTo(file.toPath)
        }
      }
    } catch {
      case e: Exception =>
        throw new Exception(
          "Error when downloading files from GCP Storage directory: " + e.getMessage)
    }
  }

}
