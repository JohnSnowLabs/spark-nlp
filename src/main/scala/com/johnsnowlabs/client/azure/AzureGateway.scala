package com.johnsnowlabs.client.azure

import com.azure.identity.DefaultAzureCredentialBuilder
import com.azure.storage.blob.models.ListBlobsOptions
import com.azure.storage.blob.{BlobServiceClient, BlobServiceClientBuilder}
import com.johnsnowlabs.client.CloudStorage
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.hadoop.fs.{FileSystem, Path}

import java.io.{File, FileOutputStream, InputStream}
import scala.jdk.CollectionConverters.asScalaIteratorConverter

class AzureGateway(storageAccountName: String) extends CloudStorage {

  private lazy val azureClient: BlobServiceClient = {
    // The default credential first checks environment variables for configuration
    // If environment configuration is incomplete, it will try managed identity
    val defaultCredential = new DefaultAzureCredentialBuilder().build()
    val blobServiceClient = new BlobServiceClientBuilder()
      .endpoint(s"https://$storageAccountName.blob.core.windows.net/")
      .credential(defaultCredential)
      .buildClient()

    blobServiceClient
  }

  override def doesBucketPathExist(bucketName: String, filePath: String): Boolean = {
    val blobClient = azureClient
      .getBlobContainerClient(bucketName)
      .getBlobClient(filePath)

    blobClient.exists()
  }

  override def copyFileToBucket(bucketName: String, destinationPath: String, inputStream: InputStream): Unit = {
    //bucketName="test", destinationPath="sentence_detector_dl_en_2.7.0_2.4_1609611052663/metadata/_SUCCESS"
    val blockBlobClient = azureClient.getBlobContainerClient(bucketName)
      .getBlobClient(destinationPath)
      .getBlockBlobClient

    val streamSize = inputStream.available()
    blockBlobClient.upload(inputStream, streamSize)
  }

  override def copyInputStreamToBucket(bucketName: String, filePath: String, sourceFilePath: String): Unit = {
    val fileSystem = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)
    val inputStream = fileSystem.open(new Path(sourceFilePath))

    val blockBlobClient = azureClient.getBlobContainerClient(bucketName)
      .getBlobClient(filePath)
      .getBlockBlobClient

    val streamSize = inputStream.available()
    blockBlobClient.upload(inputStream, streamSize)
  }

  override def downloadFilesFromBucketToDirectory(bucketName: String, filePath: String, directoryPath: String, isIndex: Boolean): Unit = {
    try {

      val blobContainerClient = azureClient.getBlobContainerClient(bucketName)

      val blobs = blobContainerClient.listBlobs(new ListBlobsOptions().setPrefix(filePath), null)
        .iterator()
        .asScala
        .toSeq

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
        throw new Exception("Error when downloading files from Azure Blob Storage: " + e.getMessage)
    }
  }
}
