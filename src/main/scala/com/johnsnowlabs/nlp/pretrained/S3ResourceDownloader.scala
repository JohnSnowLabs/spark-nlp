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

import com.johnsnowlabs.client.CloudResources
import com.johnsnowlabs.client.aws.AWSGateway
import com.johnsnowlabs.client.util.CloudHelper
import com.johnsnowlabs.util.FileHelper
import org.apache.hadoop.fs.Path
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import java.nio.file.Files
import java.util.zip.ZipInputStream
import scala.collection.mutable

class S3ResourceDownloader(
    bucket: => String,
    s3Path: => String,
    cacheFolder: => String,
    credentialsType: => String,
    region: String = "us-east-1")
    extends ResourceDownloader {

  private val logger: Logger = LoggerFactory.getLogger(this.getClass.toString)

  private val repoFolder2Metadata: mutable.Map[String, RepositoryMetadata] =
    mutable.Map[String, RepositoryMetadata]()
  val cachePath = new Path(cacheFolder)

  private val isNotCloudPath = !CloudHelper.isCloudPath(cacheFolder)

  private lazy val doesNotExistCachePath = {
    try {
      !fileSystem.exists(cachePath)
    } catch {
      case e: Exception =>
        logger.error(s"Error checking cache path existence: ${e.getMessage}")
        false
    }
  }

  if (isNotCloudPath && doesNotExistCachePath) {
    fileSystem.mkdirs(cachePath)
  }

  lazy val awsGateway = new AWSGateway(region = region, credentialsType = credentialsType)

  def downloadMetadataIfNeed(folder: String): List[ResourceMetadata] = {
    val lastMetadataState = repoFolder2Metadata.get(folder)
    val metadataFilePath = awsGateway.getS3File(s3Path, folder, "metadata.json")
    val metadataObject = awsGateway.client.getObject(bucket, metadataFilePath)
    val lastModifiedTimeInS3 = metadataObject.getObjectMetadata.getLastModified
    val needToRefresh =
      lastMetadataState.isEmpty || lastMetadataState.get.lastModified.before(lastModifiedTimeInS3)
    if (!needToRefresh) {
      metadataObject.close()
      lastMetadataState.get.metadata
    } else {
      val metadata = ResourceMetadata.readResources(metadataObject.getObjectContent)
      metadataObject.close()
      repoFolder2Metadata(folder) = RepositoryMetadata(
        folder,
        lastModifiedTimeInS3,
        java.util.Date.from(java.time.Instant.now()),
        metadata)
      metadata
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
      logger.info(s"In S3ResourceDownloader.download: $s3FilePath")
      if (!awsGateway.doesS3ObjectExist(bucket, s3FilePath)) {
        logger.info("Resource not found in S3")
        None
      } else {
        logger.info("Resource found in S3")
        val sourceS3URI = s"s3a://$bucket/$s3FilePath"
        val zipFile = sourceS3URI.split("/").last
        val modelName = zipFile.substring(0, zipFile.indexOf(".zip"))

        logger.info("Before cachePath.toString: " + cachePath.toString)
        cachePath.toString match {
          case path if CloudHelper.isCloudPath(path) => {
            logger.info(s"In S3ResourceDownloader.cachePath is cloud path: $path")
            CloudResources.downloadModelFromCloud(
              awsGateway,
              cachePath.toString,
              modelName,
              sourceS3URI)
          }
          case _ => {
            logger.info(s"In S3ResourceDownloader before downloadAndUnzipFile")
            val destinationFile = new Path(cachePath.toString, resource.fileName)
            downloadAndUnzipFile(destinationFile, resource, s3FilePath)
          }
        }
      }
    }
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
    // handle s3FilePath options:
    // 1--> s3://auxdata.johnsnowlabs.com/public/models/albert_base_sequence_classifier_ag_news_en_3.4.0_3.0_1639648298937.zip
    // 2--> public/models/albert_base_sequence_classifier_ag_news_en_3.4.0_3.0_1639648298937.zip

    val newS3FilePath = if (CloudHelper.isS3Path(s3FilePath)) {
      CloudHelper.parseS3URI(s3FilePath)._2
    } else s3FilePath

    val s3File = newS3FilePath.split("/").last

    val destinationFile = new Path(cachePath.toString + "/" + s3File)
    val splitPath = destinationFile.toString.substring(0, destinationFile.toString.length - 4)

    if (!(fileSystem.exists(destinationFile) || fileSystem.exists(new Path(splitPath)))) {
      // 1. Create tmp file
      val tmpFileName = Files.createTempFile(s3File, "").toString
      val tmpFile = new File(tmpFileName)

      val newStrfilePath: String = newS3FilePath
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

}
