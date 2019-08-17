package com.johnsnowlabs.nlp.pretrained

import java.io.File
import java.nio.file.Files
import java.sql.Timestamp
import java.util.Calendar
import java.util.zip.ZipInputStream

import com.amazonaws.auth.AWSCredentials
import com.amazonaws.regions.RegionUtils
import com.amazonaws.services.s3.AmazonS3Client
import com.amazonaws.services.s3.model.GetObjectRequest
import com.amazonaws.{AmazonServiceException, ClientConfiguration}
import com.johnsnowlabs.util.{ConfigHelper, FileHelper}
import org.apache.hadoop.fs.Path

import scala.collection.mutable


class S3ResourceDownloader(bucket: => String,
                           s3Path: => String,
                           cacheFolder: => String,
                           credentials: => Option[AWSCredentials] = None,
                           region: String = "us-east-1"
                          )
  extends ResourceDownloader with AutoCloseable {

  // repository Folder -> repository Metadata
  val repoFolder2Metadata = mutable.Map[String, RepositoryMetadata]()
  val cachePath = new Path(cacheFolder)

  if (!fs.exists(cachePath)) {
    fs.mkdirs(cachePath)
  }

  lazy val client = {
    val regionObj = RegionUtils.getRegion(region)

    val config = new ClientConfiguration()
    val timeout = ConfigHelper.getConfigValue(ConfigHelper.s3SocketTimeout).map(_.toInt).getOrElse(0)
    config.setSocketTimeout(timeout)

    val s3Client = {
      if (credentials.isDefined) {
        new AmazonS3Client(credentials.get, config)
      } else {
        new AmazonS3Client(config)
      }
    }

    s3Client.setRegion(regionObj)
    s3Client
  }

  def downloadMetadataIfNeed(folder: String): List[ResourceMetadata] = {
    val lastState = repoFolder2Metadata.get(folder)

    val fiveMinsBefore = getTimestamp(-5)
    val needToRefersh = lastState.isEmpty || lastState.get.lastMetadataDownloaded.before(fiveMinsBefore)

    if (!needToRefersh) {

      lastState.get.metadata

    }
    else {

      val metaFile = getS3File(s3Path, folder, "metadata.json")
      val obj = client.getObject(bucket, metaFile)
      val metadata = ResourceMetadata.readResources(obj.getObjectContent)
      val version = obj.getObjectMetadata.getVersionId

      RepositoryMetadata(metaFile, folder, version, getTimestamp(), metadata)

      metadata
    }
  }


  def resolveLink(request: ResourceRequest): Option[ResourceMetadata] = {
    val metadata = downloadMetadataIfNeed(request.folder)
    ResourceMetadata.resolveResource(metadata, request)
  }

  /**
    * Download resource to local file
    *
    * @param request Resource request
    * @return Downloaded file or None if resource is not found
    */
  override def download(request: ResourceRequest): Option[String] = {

    val link = resolveLink(request)
    link.flatMap {
      resource =>
        val s3FilePath = getS3File(s3Path, request.folder, resource.fileName)
        val dstFile = new Path(cachePath.toString, resource.fileName)
        val splitPath = dstFile.toString.substring(0, dstFile.toString.length - 4)
        if (!client.doesObjectExist(bucket, s3FilePath)) {
          None
        } else {
          if (!(fs.exists(dstFile) || fs.exists(new Path(splitPath)))) {

            // 1. Create tmp file
            val tmpFileName = Files.createTempFile(resource.fileName, "").toString
            val tmpFile = new File(tmpFileName)

            // 2. Download content to tmp file
            val req = new GetObjectRequest(bucket, s3FilePath)
            client.getObject(req, tmpFile)
            // 3. validate checksum
            if (!resource.checksum.equals(""))
              require(FileHelper.generateChecksum(tmpFileName).equals(resource.checksum), "Checksum validation failed!")

            // 4. Move tmp file to destination
            fs.moveFromLocalFile(new Path(tmpFile.toString), dstFile)
            println("downloading")

          }

          // 5. Unzip if needs
          if (resource.isZipped) {
            //if not already unzipped
            if (!fs.exists(new Path(splitPath))) {
              val zis = new ZipInputStream(fs.open(dstFile))
              val buf = Array.ofDim[Byte](1024)
              var entry = zis.getNextEntry
              require(dstFile.toString.substring(dstFile.toString.length - 4) == ".zip", "Not a zip file.")

              while (entry != null) {
                if (!entry.isDirectory) {
                  val entryName = new Path(splitPath, entry.getName)
                  val outputStream = fs.create(entryName)
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
              //delete the zip file
              fs.delete(dstFile, true)
            }

            Some(splitPath)

          } else {
            Some(dstFile.getName)
          }
        }
    }
  }

  override def getDownloadSize(request: ResourceRequest): Option[Long] = {

    val link = resolveLink(request)
    link.flatMap {
      resource =>
        val s3FilePath = getS3File(s3Path, request.folder, resource.fileName)
        val meta = client.getObjectMetadata(bucket, s3FilePath)
        return Some(meta.getContentLength)
    }
  }

  override def close(): Unit = {
    client.shutdown()
  }

  override def clearCache(request: ResourceRequest): Unit = {
    val metadata = downloadMetadataIfNeed(request.folder)

    val resources = ResourceMetadata.resolveResource(metadata, request)
    for (resource <- resources) {
      val fileName = new Path(cachePath.toString, resource.fileName)
      if (fs.exists(fileName))
        fs.delete(fileName, true)

      if (resource.isZipped) {
        require(fileName.toString.substring(fileName.toString.length - 4) == ".zip")
        val unzipped = fileName.toString.substring(0, fileName.toString.length - 4)
        val unzippedFile = new Path(unzipped)
        if (fs.exists(unzippedFile))
          fs.delete(unzippedFile, true)
      }
    }
  }

  private def getTimestamp(addMinutes: Int = 0): Timestamp = {
    val cal = Calendar.getInstance()
    cal.add(Calendar.MINUTE, addMinutes)
    val timestamp = new Timestamp(cal.getTime().getTime())
    cal.clear()
    timestamp
  }

  private def getS3File(parts: String*): String = {
    parts
      .map(part => part.stripSuffix("/"))
      .filter(part => part.nonEmpty)
      .mkString("/")
  }

  implicit class S3ClientWrapper(client: AmazonS3Client) {

    def doesObjectExist(bucket: String, key: String): Boolean = {
      try {
        client.getObjectMetadata(bucket, key)
        true
      } catch {
        case e: AmazonServiceException => if (e.getStatusCode == 404) return false else throw e
      }
    }
  }

}
