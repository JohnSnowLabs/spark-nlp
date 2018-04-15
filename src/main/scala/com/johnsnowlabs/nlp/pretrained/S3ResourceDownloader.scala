package com.johnsnowlabs.nlp.pretrained

import java.io.File
import java.nio.file.{Files, Paths}
import java.sql.Timestamp
import java.util.Calendar

import com.amazonaws.ClientConfiguration
import com.amazonaws.auth.{AWSCredentials, AWSStaticCredentialsProvider, BasicAWSCredentials}
import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.amazonaws.services.s3.model.GetObjectRequest
import com.johnsnowlabs.util.{FileHelper, ZipArchiveUtil}
import org.apache.commons.io.FileUtils

import scala.collection.mutable


class S3ResourceDownloader(bucket: String,
                           s3Path: String,
                           cacheFolder: String,
                           credentials: Option[AWSCredentials] = None,
                           region: String = "us-east-1"
                          )
  extends ResourceDownloader with AutoCloseable {

  // repository Folder -> repository Metadata
  val repoFolder2Metadata = mutable.Map[String, RepositoryMetadata]()

   if (!new File(cacheFolder).exists()) {
    FileUtils.forceMkdir(new File(cacheFolder))
  }

  lazy val client = {

    val builder = AmazonS3ClientBuilder.standard()
    if (credentials.isDefined)
      builder.setCredentials(new AWSStaticCredentialsProvider(credentials.get))

    builder.setRegion(region)
    val config = new ClientConfiguration()
    //config.setSocketTimeout(0)
    //config.setConnectionTimeout(0)
    //config.setMaxErrorRetry(20)
    //config.setMaxConnections(500)
    //config.setUseTcpKeepAlive(true)
    //config.setRequestTimeout(2000000)

    builder.setClientConfiguration(config)

    builder.build()
  }


  private def downloadMetadataIfNeed(folder: String): List[ResourceMetadata] = {
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
    * @param request        Resource request
    * @return               Downloaded file or None if resource is not found
    */
  override def download(request: ResourceRequest): Option[String] = {

    val link = resolveLink(request)
    link.flatMap {
      resource =>
        val s3FilePath = getS3File(s3Path, request.folder, resource.fileName)
        val dstFile = new File(cacheFolder, resource.fileName)
        if (!client.doesObjectExist(bucket, s3FilePath)) {
          None
        } else {
          if (!dstFile.exists()) {

            //val obj = client.getObject(bucket, s3FilePath)
            // 1. Create tmp file
            val tmpFileName = Files.createTempFile(resource.fileName, "").toString
            val tmpFile = new File(tmpFileName)

            // 2. Download content to tmp file
            val req = new GetObjectRequest(bucket, s3FilePath)
            client.getObject(req, tmpFile)
            //FileUtils.copyInputStreamToFile(obj.getObjectContent, tmpFile)

            // 3. Move tmp file to destination
            FileUtils.moveFile(tmpFile, dstFile)
          }

          // 4. Unzip if needs
          val dstFileName = if (resource.isZipped)
            ZipArchiveUtil.unzip(dstFile)
          else
            dstFile.getPath

          Some(dstFileName)
        }
    }
  }

  override def close(): Unit = {
    client.shutdown()
  }

  override def clearCache(request: ResourceRequest): Unit = {
    val metadata = downloadMetadataIfNeed(request.folder)

    val resources = ResourceMetadata.resolveResource(metadata, request)
    for (resource <- resources) {
      val fileName = Paths.get(cacheFolder, resource.fileName).toString
      val file = new File(fileName)
      if (file.exists()){
        file.delete()
      }

      if (resource.isZipped) {
        require(fileName.substring(fileName.length - 4) == ".zip")
        val unzipped = fileName.substring(0, fileName.length - 4)
        val unzippedFile = new File(unzipped)
        if (unzippedFile.exists()) {
          FileHelper.delete(unzippedFile.toPath.toString)
        }
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
}
