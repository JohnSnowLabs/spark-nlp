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

package com.johnsnowlabs.nlp.pretrained

import com.amazonaws.services.s3.model.ObjectMetadata
import com.johnsnowlabs.client.aws.AWSGateway
import com.johnsnowlabs.client.gcp.GCPGateway
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{ConfigHelper, FileHelper}
import org.apache.commons.io.IOUtils
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.SparkSession

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, File}
import java.nio.file.Files
import java.sql.Timestamp
import java.util.Calendar
import java.util.zip.ZipInputStream
import scala.collection.mutable

class S3ResourceDownloader(
    bucket: => String,
    s3Path: => String,
    cacheFolder: => String,
    credentialsType: => String,
    region: String = "us-east-1")
    extends ResourceDownloader {

  private val repoFolder2Metadata: mutable.Map[String, RepositoryMetadata] =
    mutable.Map[String, RepositoryMetadata]()
  val cachePath = new Path(cacheFolder)

  if (!doesCacheFolderInCloud && !fileSystem.exists(cachePath)) {
    fileSystem.mkdirs(cachePath)
  }

  lazy val awsGateway = new AWSGateway(region = region, credentialsType = credentialsType)

  private def doesCacheFolderInCloud(): Boolean = {
    cacheFolder.startsWith("s3") || cacheFolder.startsWith("gs")
  }

  def downloadMetadataIfNeed(folder: String): List[ResourceMetadata] = {
    val lastState = repoFolder2Metadata.get(folder)

    val fiveMinsBefore = getTimestamp(-5)
    val needToRefresh = lastState.isEmpty || lastState.get.lastMetadataDownloaded
      .before(fiveMinsBefore)

    if (!needToRefresh) {
      lastState.get.metadata
    } else {
      awsGateway.getMetadata(s3Path, folder, bucket)
    }
  }

  def resolveLink(request: ResourceRequest): Option[ResourceMetadata] = {
    val metadata = downloadMetadataIfNeed(request.folder)
    ResourceMetadata.resolveResource(metadata, request)
  }

  /** Download resource to local file
    *
    * @param request
    *   Resource request
    * @return
    *   Downloaded file or None if resource is not found
    */
  override def download(request: ResourceRequest): Option[String] = {
    val link = resolveLink(request)
    link.flatMap { resource =>
      val s3FilePath = awsGateway.getS3File(s3Path, request.folder, resource.fileName)
      if (!awsGateway.doesS3ObjectExist(bucket, s3FilePath)) {
        None
      } else {

        val s3Path = "^s3.*".r
        val gcpStoragePath = "^gs.*".r

        val sourceS3URI = s"s3a://$bucket/$s3FilePath"
        val zipFile = sourceS3URI.split("/").last
        val modelName = zipFile.substring(0, zipFile.indexOf(".zip"))

        cachePath.toString match {
          case s3Path() => {
            val destinationS3URI = cachePath.toString.replace("s3:", "s3a:")
            val modelExists =
              doesModelExistInExternalCloudStorage(modelName, destinationS3URI, "S3")

            if (!modelExists) {
              val destinationKey = unzipInExternalCloudStorage(
                ResourceHelper.spark,
                sourceS3URI,
                destinationS3URI,
                "S3")
              Option(destinationKey)
            } else {
              Option(destinationS3URI + "/" + modelName)
            }

          }
          case gcpStoragePath() => {
            val sourceS3URI = s"s3a://$bucket/$s3FilePath"

            val modelExists =
              doesModelExistInExternalCloudStorage(modelName, cachePath.toString, "GCP")

            if (!modelExists) {
              val destination = unzipInExternalCloudStorage(
                ResourceHelper.spark,
                sourceS3URI,
                cachePath.toString,
                "GCP")
              Option(destination)
            } else {
              Option(cachePath.toString + "/" + modelName)
            }

          }
          case _ => {
            val destinationFile = new Path(cachePath.toString, resource.fileName)
            downloadAndUnzipFile(destinationFile, resource, s3FilePath)
          }
        }
      }
    }
  }

  private def doesModelExistInExternalCloudStorage(
      modelName: String,
      destinationURI: String,
      destinationCloud: String): Boolean = {

    destinationCloud match {
      case "S3" => {
        val (accessKeyId, secretKey, sessionToken) = ConfigHelper.getHadoopS3Config
        val awsDestinationGateway = new AWSGateway(accessKeyId, secretKey, sessionToken)
        val (destinationBucketName, destinationKey) = ResourceHelper.parseS3URI(destinationURI)

        val modelPath = destinationKey + "/" + modelName

        awsDestinationGateway.doesS3FolderExist(destinationBucketName, modelPath)
      }
      case "GCP" => {
        val (gcpGateway, destinationBucketName, destinationStoragePath) = getGCPStorageConfig(
          destinationURI)
        val modelPath = destinationStoragePath + "/" + modelName

        gcpGateway.doesFolderExist(destinationBucketName, modelPath)
      }
    }

  }

  private def unzipInExternalCloudStorage(
      sparkSession: SparkSession,
      sourceS3URI: String,
      destinationStorageURI: String,
      destinationCloud: String) = {

    val (sourceBucketName, sourceKey) = ResourceHelper.parseS3URI(sourceS3URI)
    val zippedModel = awsGateway.getS3Object(sourceBucketName, sourceKey)
    val zipInputStream = new ZipInputStream(zippedModel.getObjectContent)
    var zipEntry = zipInputStream.getNextEntry

    val zipFile = sourceKey.split("/").last
    val modelName = zipFile.substring(0, zipFile.indexOf(".zip"))

    println(s"Uploading model $modelName to external Cloud Storage URI: $destinationStorageURI")
    while (zipEntry != null) {
      if (!zipEntry.isDirectory) {
        val outputStream = new ByteArrayOutputStream()
        IOUtils.copy(zipInputStream, outputStream)
        val inputStream = new ByteArrayInputStream(outputStream.toByteArray)

        if (destinationCloud == "S3") {
          val (awsGatewayDestination, destinationBucketName, destinationKey) =
            getS3Config(sparkSession, destinationStorageURI)
          val fileName = s"$modelName/${zipEntry.getName}"
          val destinationS3Path = destinationKey + "/" + fileName

          awsGatewayDestination.client.putObject(
            destinationBucketName,
            destinationS3Path,
            inputStream,
            new ObjectMetadata())

        } else {
          val (gcpGateway, destinationBucketName, destinationStoragePath) = getGCPStorageConfig(
            destinationStorageURI)
          val destinationGCPStoragePath =
            s"$destinationStoragePath/$modelName/${zipEntry.getName}"

          gcpGateway.copyFileToGCPStorage(
            destinationBucketName,
            destinationGCPStoragePath,
            inputStream)
        }

      }
      zipEntry = zipInputStream.getNextEntry
    }
    destinationStorageURI + "/" + modelName
  }

  private def getS3Config(sparkSession: SparkSession, destinationS3URI: String) = {
    var accessKeyId =
      sparkSession.sparkContext.hadoopConfiguration.get("fs.s3a.access.key")
    var secretAccessKey =
      sparkSession.sparkContext.hadoopConfiguration.get("fs.s3a.secret.key")
    var sessionToken =
      sparkSession.sparkContext.hadoopConfiguration.get("fs.s3a.session.token")

    if (accessKeyId == null) accessKeyId = ""
    if (secretAccessKey == null) secretAccessKey = ""
    if (sessionToken == null) sessionToken = ""

    if (accessKeyId == "" && secretAccessKey == "") {
      throw new IllegalAccessException(
        "Using S3 as cachePath requires to define access.key and secret.key hadoop configuration")
    }
    val awsGatewayDestination = new AWSGateway(accessKeyId, secretAccessKey, sessionToken)

    val (destinationBucketName, destinationKey) = ResourceHelper.parseS3URI(destinationS3URI)

    (awsGatewayDestination, destinationBucketName, destinationKey)
  }

  private def getGCPStorageConfig(destinationGCPStorageURI: String) = {
    val gcpGateway = new GCPGateway()
    val (destinationBucketName, destinationStoragePath) =
      ResourceHelper.parseGCPStorageURI(destinationGCPStorageURI)

    (gcpGateway, destinationBucketName, destinationStoragePath)
  }

  def downloadAndUnzipFile(
      destinationFile: Path,
      resource: ResourceMetadata,
      s3FilePath: String): Option[String] = {

    val splitPath = destinationFile.toString.substring(0, destinationFile.toString.length - 4)
    if (!(fileSystem.exists(destinationFile) || fileSystem.exists(new Path(splitPath)))) {
      // 1. Create tmp file
      val tmpFileName = Files.createTempFile(resource.fileName, "").toString
      val tmpFile = new File(tmpFileName)

      // 2. Download content to tmp file
      awsGateway.getS3Object(bucket, s3FilePath, tmpFile)
      // 3. validate checksum
      if (!resource.checksum.equals(""))
        require(
          FileHelper.generateChecksum(tmpFileName).equals(resource.checksum),
          "Checksum validation failed!")

      // 4. Move tmp file to destination
      fileSystem.moveFromLocalFile(new Path(tmpFile.toString), destinationFile)

    }

    // 5. Unzip if needs
    if (resource.isZipped) {
      // if not already unzipped
      if (!fileSystem.exists(new Path(splitPath))) {
        val zis = new ZipInputStream(fileSystem.open(destinationFile))
        val buf = Array.ofDim[Byte](1024)
        var entry = zis.getNextEntry
        require(
          destinationFile.toString.substring(destinationFile.toString.length - 4) == ".zip",
          "Not a zip file.")

        while (entry != null) {
          if (!entry.isDirectory) {
            val entryName = new Path(splitPath, entry.getName)
            val outputStream = fileSystem.create(entryName)
            var bytesRead = zis.read(buf, 0, 1024)
            while (bytesRead > -1) {
              outputStream.write(buf, 0, bytesRead)
              bytesRead = zis.read(buf, 0, 1024)
            }
            outputStream.close()
          }
          zis.closeEntry()
          entry = zis.getNextEntry
        }
        zis.close()
        // delete the zip file
        fileSystem.delete(destinationFile, true)
      }
      Some(splitPath)
    } else {
      Some(destinationFile.getName)
    }
  }

  def downloadAndUnzipFile(s3FilePath: String, unzip: Boolean): Option[String] = {

    val s3File = s3FilePath.split("/").last
    val destinationFile = new Path(cachePath.toString + "/" + s3File)
    val splitPath = destinationFile.toString.substring(0, destinationFile.toString.length - 4)

    if (!(fileSystem.exists(destinationFile) || fileSystem.exists(new Path(splitPath)))) {
      // 1. Create tmp file
      val tmpFileName = Files.createTempFile(s3File, "").toString
      val tmpFile = new File(tmpFileName)

      val newStrfilePath: String = s3FilePath
      val mybucket: String = bucket
      // 2. Download content to tmp file
      awsGateway.getS3Object(mybucket, newStrfilePath, tmpFile)

      // 4. Move tmp file to destination
      fileSystem.moveFromLocalFile(new Path(tmpFile.toString), destinationFile)
    }
    if (unzip) {
      if (!fileSystem.exists(new Path(splitPath))) {
        val zis = new ZipInputStream(fileSystem.open(destinationFile))
        val buf = Array.ofDim[Byte](1024)
        var entry = zis.getNextEntry
        require(
          destinationFile.toString.substring(destinationFile.toString.length - 4) == ".zip",
          "Not a zip file.")

        while (entry != null) {
          if (!entry.isDirectory) {
            val entryName = new Path(splitPath, entry.getName)
            val outputStream = fileSystem.create(entryName)
            var bytesRead = zis.read(buf, 0, 1024)
            while (bytesRead > -1) {
              outputStream.write(buf, 0, bytesRead)
              bytesRead = zis.read(buf, 0, 1024)
            }
            outputStream.close()
          }
          zis.closeEntry()
          entry = zis.getNextEntry
        }
        zis.close()
        // delete the zip file
        fileSystem.delete(destinationFile, true)
      }
    }
    Some(splitPath)

  }

  override def getDownloadSize(request: ResourceRequest): Option[Long] = {
    val link = resolveLink(request)
    link.flatMap { resource =>
      awsGateway.getS3DownloadSize(s3Path, request.folder, resource.fileName, bucket)
    }
  }

  override def clearCache(request: ResourceRequest): Unit = {
    val metadata = downloadMetadataIfNeed(request.folder)

    val resources = ResourceMetadata.resolveResource(metadata, request)
    for (resource <- resources) {
      val fileName = new Path(cachePath.toString, resource.fileName)
      if (fileSystem.exists(fileName))
        fileSystem.delete(fileName, true)

      if (resource.isZipped) {
        require(fileName.toString.substring(fileName.toString.length - 4) == ".zip")
        val unzipped = fileName.toString.substring(0, fileName.toString.length - 4)
        val unzippedFile = new Path(unzipped)
        if (fileSystem.exists(unzippedFile))
          fileSystem.delete(unzippedFile, true)
      }
    }
  }

  private def getTimestamp(addMinutes: Int = 0): Timestamp = {
    val cal = Calendar.getInstance()
    cal.add(Calendar.MINUTE, addMinutes)
    val timestamp = new Timestamp(cal.getTime.getTime)
    cal.clear()
    timestamp
  }

}
