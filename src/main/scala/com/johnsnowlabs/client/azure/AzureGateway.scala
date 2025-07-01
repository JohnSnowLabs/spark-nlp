package com.johnsnowlabs.client.azure

import com.azure.storage.blob.models.ListBlobsOptions
import com.azure.storage.blob.{BlobServiceClient, BlobServiceClientBuilder}
import com.johnsnowlabs.client.CloudStorage
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.IOUtils

import java.io._
import scala.jdk.CollectionConverters.asScalaIteratorConverter

class AzureGateway(
    storageAccountName: String,
    accountKey: String,
    isFabricLakehouse: Boolean = false)
    extends CloudStorage {

  private lazy val blobServiceClient: BlobServiceClient = {
    val connectionString =
      s"DefaultEndpointsProtocol=https;AccountName=$storageAccountName;AccountKey=$accountKey;EndpointSuffix=core.windows.net"

    val blobServiceClient = new BlobServiceClientBuilder()
      .connectionString(connectionString)
      .buildClient()

    blobServiceClient
  }

  private def getHadoopFS(path: String): FileSystem = {
    val uri = new java.net.URI(path)
    FileSystem.get(uri, ResourceHelper.spark.sparkContext.hadoopConfiguration)
  }

  override def doesBucketPathExist(bucketName: String, filePath: String): Boolean = {
    if (isFabricLakehouse) {
      doesPathExistAbfss(bucketName)
    } else {
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
  }

  override def copyFileToBucket(
      bucketName: String,
      destinationPath: String,
      inputStream: InputStream): Unit = {
    if (isFabricLakehouse) {
      copyInputStreamToAbfssUri(bucketName, inputStream)
    } else {
      val blockBlobClient = blobServiceClient
        .getBlobContainerClient(bucketName)
        .getBlobClient(destinationPath)
        .getBlockBlobClient

      val streamSize = inputStream.available()
      blockBlobClient.upload(inputStream, streamSize)
    }
  }

  override def copyInputStreamToBucket(
      bucketName: String,
      filePath: String,
      sourceFilePath: String): Unit = {
    if (isFabricLakehouse) {
      val abfssPath = s"abfss://$bucketName/$filePath"
      val fs = getHadoopFS(abfssPath)
      val inputStream = fs.open(new Path(sourceFilePath))
      val outputStream = fs.create(new Path(abfssPath), true)
      IOUtils.copyBytes(inputStream, outputStream, 4096, true)
    } else {
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
  }

  override def downloadFilesFromBucketToDirectory(
      bucketName: String,
      filePath: String,
      directoryPath: String,
      isIndex: Boolean): Unit = {
    if (isFabricLakehouse) {
      downloadFilesFromAbfssUri(bucketName, directoryPath)
    } else {
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

  /** Download all files under abfss URI to local directory (Fabric) */
  private def downloadFilesFromAbfssUri(uri: String, directory: String): Unit = {
    val fs =
      FileSystem.get(new Path(uri).toUri, ResourceHelper.spark.sparkContext.hadoopConfiguration)
    val files = fs.globStatus(new Path(uri + "/*"))
    if (files == null || files.isEmpty) throw new Exception(s"No files found at $uri")
    files.foreach { status =>
      val fileName = status.getPath.getName
      val localFile = new File(s"$directory/$fileName")
      if (status.isDirectory) {
        localFile.mkdirs()
      } else {
        localFile.getParentFile.mkdirs()
        val out = new FileOutputStream(localFile)
        val in = fs.open(status.getPath)
        IOUtils.copyBytes(in, out, 4096, true)
        out.close()
        in.close()
      }
    }
  }

  private def doesPathExistAbfss(uri: String): Boolean = {
    val path = new Path(uri)
    val fs = FileSystem.get(path.toUri, ResourceHelper.spark.sparkContext.hadoopConfiguration)
    fs.exists(path)
  }

  private def copyInputStreamToAbfssUri(uri: String, in: InputStream): Unit = {
    val path = new Path(uri)
    val fs = FileSystem.get(path.toUri, ResourceHelper.spark.sparkContext.hadoopConfiguration)
    val out = fs.create(path, true)
    try {
      IOUtils.copyBytes(in, out, 4096, true)
    } finally {
      out.close()
      in.close()
    }
  }

}
