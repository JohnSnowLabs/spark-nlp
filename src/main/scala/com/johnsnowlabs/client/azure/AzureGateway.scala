package com.johnsnowlabs.client.azure

import com.azure.storage.blob.models.ListBlobsOptions
import com.azure.storage.blob.{BlobServiceClient, BlobServiceClientBuilder}
import com.johnsnowlabs.client.CloudStorage
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.IOUtils

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, File, FileOutputStream, InputStream}
import scala.jdk.CollectionConverters.asScalaIteratorConverter

class AzureGateway(storageAccountName: String, accountKey: String) extends CloudStorage {

  private lazy val blobServiceClient: BlobServiceClient = {
    val connectionString =
      s"DefaultEndpointsProtocol=https;AccountName=$storageAccountName;AccountKey=$accountKey;EndpointSuffix=core.windows.net"

    val blobServiceClient = new BlobServiceClientBuilder()
      .connectionString(connectionString)
      .buildClient()

    blobServiceClient
  }

  override def doesBucketPathExist(bucketName: String, filePath: String): Boolean = {
    val blobContainerClient = blobServiceClient
      .getBlobContainerClient(bucketName)

    val prefix = if (filePath.endsWith("/")) filePath else filePath + "/"

    val blobs = blobContainerClient
      .listBlobs()
      .iterator()
      .asScala
      .filter(_.getName.startsWith(prefix))

    blobs.nonEmpty
  }

  override def copyFileToBucket(
      bucketName: String,
      destinationPath: String,
      inputStream: InputStream): Unit = {

    val blockBlobClient = blobServiceClient
      .getBlobContainerClient(bucketName)
      .getBlobClient(destinationPath)
      .getBlockBlobClient

    val streamSize = inputStream.available()
    blockBlobClient.upload(inputStream, streamSize)
  }

  override def copyInputStreamToBucket(
      bucketName: String,
      filePath: String,
      sourceFilePath: String): Unit = {
    val fileSystem = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)
    val inputStream = fileSystem.open(new Path(sourceFilePath))

    val byteArrayOutputStream = new ByteArrayOutputStream()
    IOUtils.copyBytes(inputStream, byteArrayOutputStream, 4096, true)

    val byteArrayInputStream = new ByteArrayInputStream(byteArrayOutputStream.toByteArray)

    val blockBlobClient = blobServiceClient
      .getBlobContainerClient(bucketName)
      .getBlobClient(filePath)
      .getBlockBlobClient

    val streamSize = byteArrayInputStream.available()
    blockBlobClient.upload(byteArrayInputStream, streamSize)
  }

  override def downloadFilesFromBucketToDirectory(
      bucketName: String,
      filePath: String,
      directoryPath: String,
      isIndex: Boolean): Unit = {
    try {
      val blobContainerClient = blobServiceClient.getBlobContainerClient(bucketName)
      val blobOptions = new ListBlobsOptions().setPrefix(filePath)
      val blobs = blobContainerClient
        .listBlobs(blobOptions, null)
        .iterator()
        .asScala
        .toSeq

      if (blobs.isEmpty) {
        throw new Exception(
          s"Not found blob path $filePath in container $bucketName when downloading files from Azure Blob Storage")
      }

      blobs.foreach { blobItem =>
        val blobName = blobItem.getName
        val blobClient = blobContainerClient.getBlobClient(blobName)

        val file = new File(s"$directoryPath/$blobName")
        if (blobName.endsWith("/")) {
          file.mkdirs()
        } else {
          file.getParentFile.mkdirs()
          val outputStream = new FileOutputStream(file)
          blobClient.downloadStream(outputStream)
          outputStream.close()
        }
      }
    } catch {
      case e: Exception =>
        throw new Exception(
          "Error when downloading files from Azure Blob Storage: " + e.getMessage)
    }
  }
}
