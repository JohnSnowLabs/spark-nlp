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
import com.johnsnowlabs.client.CloudResources
import com.johnsnowlabs.client.aws.AWSGateway
import com.johnsnowlabs.client.util.CloudHelper
import com.johnsnowlabs.util.FileHelper
import org.apache.hadoop.fs.Path
import org.slf4j.{Logger, LoggerFactory}

import java.io.{EOFException, File, FileInputStream, IOException}
import java.nio.file.Files
import java.util.UUID
import java.util.concurrent.TimeUnit
import java.util.zip.{ZipException, ZipInputStream}
import scala.collection.mutable
import scala.util.Try
import scala.util.control.NonFatal

class S3ResourceDownloader(
    bucket: => String,
    s3Path: => String,
    cacheFolder: => String,
    credentialsType: => String,
    region: String = "us-east-1")
    extends ResourceDownloader {

  private val logger: Logger = LoggerFactory.getLogger(this.getClass.toString)

  // was 1024; unzip of a large model otherwise costs ~650k read/write pairs
  private val UnzipBufferSize = 65536

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

    // A HEAD is enough to decide freshness. Opening the object here would begin streaming the
    // whole index (tens of MB) only to abort it whenever the cached copy is still current.
    val indexMetadata = awsGateway
      .getS3ObjectMetadata(bucket, metadataFilePath)
      .getOrElse(throw new Exception(s"Metadata file not found in S3: $metadataFilePath"))
    val lastModifiedTimeInS3 = indexMetadata.getLastModified
    val needToRefresh = S3ResourceDownloader.needToRefresh(
      lastMetadataState.map(_.lastModified),
      lastModifiedTimeInS3)

    if (!needToRefresh) {
      lastMetadataState.get.metadata
    } else {
      val tmpFile = Files.createTempFile("sparknlp_metadata", ".json").toFile
      val metadata =
        try {
          awsGateway.getS3Object(
            bucket,
            metadataFilePath,
            tmpFile,
            Some(indexMetadata),
            showProgress = false)
          val stream = new FileInputStream(tmpFile)
          try ResourceMetadata.readResources(stream)
          finally stream.close()
        } finally {
          tmpFile.delete()
        }
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
      val objectMetadata = awsGateway.getS3ObjectMetadata(bucket, s3FilePath)
      if (objectMetadata.isEmpty) {
        logger.info("Resource not found in S3")
        None
      } else {
        logger.info("Resource found in S3")
        val sourceS3URI = s"s3a://$bucket/$s3FilePath"
        val zipFile = sourceS3URI.split("/").last
        val modelName = zipFile.substring(0, zipFile.indexOf(".zip"))

        cachePath.toString match {
          case path if CloudHelper.isCloudPath(path) => {
            logger.info(s"Downloading from cloud path: $path")
            CloudResources.downloadModelFromCloud(
              awsGateway,
              cachePath.toString,
              modelName,
              sourceS3URI)
          }
          case _ => {
            val destinationFile = new Path(cachePath.toString, resource.fileName)
            downloadAndUnzipFile(destinationFile, resource, s3FilePath, objectMetadata)
          }
        }
      }
    }
  }

  def downloadAndUnzipFile(
      destinationFile: Path,
      resource: ResourceMetadata,
      s3FilePath: String,
      objectMetadata: Option[ObjectMetadata]): Option[String] = {

    val splitPath = destinationFile.toString.substring(0, destinationFile.toString.length - 4)
    if (!(fileSystem.exists(destinationFile) || fileSystem.exists(new Path(splitPath)))) {
      // 1. Create tmp file
      val tmpFileName = Files.createTempFile(resource.fileName, "").toString
      val tmpFile = new File(tmpFileName)
      try {
        // 2. Download content to tmp file
        awsGateway.getS3Object(bucket, s3FilePath, tmpFile, objectMetadata)
        // 3. validate checksum
        if (!resource.checksum.equals("")) {
          require(
            FileHelper.generateChecksum(tmpFileName).equals(resource.checksum),
            "Checksum validation failed!")
          logger.info(s"Checksum verified for ${resource.fileName}")
        }
        // 4. Move tmp file to destination
        moveIntoCache(tmpFile, destinationFile)
      } finally {
        tmpFile.delete()
      }
    }

    // 5. Unzip if needs
    if (resource.isZipped) {
      ensureUnzipped(destinationFile, splitPath)
      Some(splitPath)
    } else {
      Some(destinationFile.getName)
    }
  }

  /** Kept with its 7.0 signature for callers built against it. */
  def downloadAndUnzipFile(
      destinationFile: Path,
      resource: ResourceMetadata,
      s3FilePath: String): Option[String] =
    downloadAndUnzipFile(destinationFile, resource, s3FilePath, None)

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
      try {
        // 2. Download content to tmp file
        // Shown: this serves Python's downloadModelDirectly, which draws the progress.
        awsGateway.getS3Object(bucket, newS3FilePath, tmpFile, None, showProgress = true)
        // 4. Move tmp file to destination
        moveIntoCache(tmpFile, destinationFile)
      } finally {
        tmpFile.delete()
      }
    }
    if (unzip) ensureUnzipped(destinationFile, splitPath)
    Some(splitPath)

  }

  /** Puts a downloaded file into the cache under its final name with one rename, so the cache
    * never holds a half-copied zip. Each install copies into a part file of its own, so two
    * installers of one model never write the same file.
    */
  private def moveIntoCache(localFile: File, destination: Path): Unit = {
    reclaimStaleParts(destination)
    val part = new Path(s"$destination.part-${UUID.randomUUID()}")
    try {
      fileSystem.moveFromLocalFile(new Path(localFile.toString), part)
      // Losing the rename to another installer of the same model is a success: only a rename
      // ever puts a file at `destination`, so whatever is there is a complete copy.
      if (!fileSystem.rename(part, destination) && !fileSystem.exists(destination))
        throw new IOException(s"Could not move $part to $destination")
    } finally {
      if (fileSystem.exists(part)) fileSystem.delete(part, false)
    }
  }

  /** Deletes part files of `destination` left by runs that died mid-copy. A live copy keeps
    * writing to its own part, so only parts untouched for `StalePartAgeMillis` are taken.
    */
  private def reclaimStaleParts(destination: Path): Unit = {
    val prefix = destination.getName + ".part-"
    val cutoff = System.currentTimeMillis() - S3ResourceDownloader.StalePartAgeMillis
    Try(fileSystem.listStatus(destination.getParent)).toOption.toSeq.flatten
      .filter(status =>
        status.getPath.getName.startsWith(prefix) && status.getModificationTime < cutoff)
      .foreach(status => Try(fileSystem.delete(status.getPath, false)))
  }

  /** Unzips the model unless another installer already put it in place, in which case a zip this
    * install just moved into the cache is redundant and goes.
    */
  private def ensureUnzipped(zipFile: Path, splitPath: String): Unit =
    if (!fileSystem.exists(new Path(splitPath))) unzipIntoCache(zipFile, splitPath)
    else if (fileSystem.exists(zipFile)) fileSystem.delete(zipFile, false)

  /** Unzips into a sibling staging directory and renames it into place, so a directory at
    * `splitPath` is always a whole model. An archive that ZipInputStream rejects (a bad entry
    * CRC, data that ends mid-entry) takes the zip with it, so the next request downloads again
    * instead of loading half a model.
    */
  private def unzipIntoCache(zipFile: Path, splitPath: String): Unit = {
    require(zipFile.getName.endsWith(".zip"), "Not a zip file.")
    val staging = new Path(s"$splitPath.tmp-${UUID.randomUUID()}")
    try {
      fileSystem.mkdirs(staging)
      val zis = new ZipInputStream(fileSystem.open(zipFile))
      try {
        val buf = Array.ofDim[Byte](UnzipBufferSize)
        var entry = zis.getNextEntry
        while (entry != null) {
          if (!entry.isDirectory) {
            val outputStream = fileSystem.create(new Path(staging, entry.getName))
            try {
              var bytesRead = zis.read(buf, 0, UnzipBufferSize)
              while (bytesRead > -1) {
                outputStream.write(buf, 0, bytesRead)
                bytesRead = zis.read(buf, 0, UnzipBufferSize)
              }
            } finally outputStream.close()
          }
          zis.closeEntry()
          entry = zis.getNextEntry
        }
      } finally zis.close()
      // Another process may have finished the same model meanwhile; theirs is as good as ours.
      val target = new Path(splitPath)
      if (!fileSystem.exists(target) && !fileSystem.rename(staging, target))
        throw new IOException(s"Could not move $staging to $splitPath")
    } catch {
      // Another installer finished the same model first; theirs is complete, ours not needed.
      case NonFatal(_) if fileSystem.exists(new Path(splitPath)) =>
      // The archive itself is bad: drop it so the next request downloads a fresh copy. Any
      // other failure leaves the zip in place for the next attempt to unpack.
      case e @ (_: ZipException | _: EOFException) =>
        fileSystem.delete(zipFile, false)
        throw e
    } finally {
      if (fileSystem.exists(staging)) fileSystem.delete(staging, true)
    }
    // delete the zip file
    fileSystem.delete(zipFile, true)
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

object S3ResourceDownloader {

  /** A part file untouched this long belongs to a run that died mid-copy. */
  private[pretrained] val StalePartAgeMillis: Long = TimeUnit.HOURS.toMillis(1)

  /** Whether a repository's cached index must be fetched from S3 again. */
  private[pretrained] def needToRefresh(
      cachedLastModified: Option[java.util.Date],
      lastModifiedInS3: java.util.Date): Boolean =
    cachedLastModified.forall(_.before(lastModifiedInS3))
}
