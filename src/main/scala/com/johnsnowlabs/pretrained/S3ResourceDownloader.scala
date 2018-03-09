package com.johnsnowlabs.pretrained

import java.io.File
import java.nio.file.{Files, Paths}
import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.johnsnowlabs.util.{Version, ZipArchiveUtil}
import org.apache.commons.io.FileUtils


class S3ResourceDownloader(bucket: String,
                           s3Path: String,
                           cacheFolder: String,
                           region: String = "us-east-1")
  extends ResourceDownloader with AutoCloseable {

  var lastMetadataVersion: Option[String] = None
  var metadata = List.empty[ResourceMetadata]
  val metadataFile = Paths.get(s3Path, "metadata.json").toString.replaceAllLiterally("\\", "/")

   if (!new File(cacheFolder).exists()) {
    FileUtils.forceMkdir(new File(cacheFolder))
  }

  lazy val client = {
    val builder = AmazonS3ClientBuilder.standard()
    builder.setRegion(region)
    builder.build()
  }

  private def downloadMetadataIfNeed(): Unit = {
    val obj = client.getObject(bucket, metadataFile)
    if (lastMetadataVersion.isEmpty || obj.getObjectMetadata.getVersionId != lastMetadataVersion.get) {
      metadata = ResourceMetadata.readResources(obj.getObjectContent)
      lastMetadataVersion = Some(obj.getObjectMetadata.getVersionId)
    }
  }

  def resolveLink(name: String,
                  language: Option[String],
                  libVersion: Version,
                  sparkVersion: Version): Option[ResourceMetadata] = {

    downloadMetadataIfNeed()
    ResourceMetadata.resolveResource(metadata, name, language, libVersion, sparkVersion)
  }

  /**
    * Download resource to local file
    *
    * @param name         Resource Name. ner_fast for example
    * @param language     Language of the model
    * @param libVersion   spark-nlp library version
    * @param sparkVersion spark library version
    * @return             downloaded file or None if resource is not found
    */
  override def download(name: String,
                        language: Option[String],
                        libVersion: Version,
                        sparkVersion: Version): Option[String] = {

    val link = resolveLink(name, language, libVersion, sparkVersion)
    link.flatMap {
      resource =>
        val s3FilePath = Paths.get(s3Path, resource.fileName).toString.replaceAllLiterally("\\", "/")
        val dstFile = new File(cacheFolder, resource.fileName)
        if (!client.doesObjectExist(bucket, s3FilePath)) {
          None
        } else {
          if (!dstFile.exists()) {
            val obj = client.getObject(bucket, s3FilePath)
            // 1. Create tmp file
            val tmpFileName = Files.createTempFile(resource.fileName, "").toString
            val tmpFile = new File(tmpFileName)

            // 2. Download content to tmp file
            FileUtils.copyInputStreamToFile(obj.getObjectContent, tmpFile)

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

  override def clearCache(name: String, language: Option[String], libVersion: Version, sparkVersion: Version): Unit = {

    val resources = ResourceMetadata.resolveResource(metadata, name, language, libVersion, sparkVersion)
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
          FileUtils.deleteDirectory(unzippedFile)
        }
      }
    }
  }
}
