package com.johnsnowlabs.client.aws

import com.amazonaws.{AmazonServiceException, ClientConfiguration}
import com.amazonaws.auth.{AWSCredentials, AWSStaticCredentialsProvider}
import com.amazonaws.services.pi.model.InvalidArgumentException
import com.amazonaws.services.s3.model.{GetObjectRequest, ObjectMetadata, PutObjectResult}
import com.amazonaws.services.s3.{AmazonS3, AmazonS3ClientBuilder}
import com.johnsnowlabs.client.CredentialParams
import com.johnsnowlabs.nlp.pretrained.ResourceMetadata
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.hadoop.fs.{FileSystem, Path}

import java.io.File

class AWSGateway(accessKeyId: String, secretAccessKey: String, sessionToken: String,
                 awsProfile: String, region: String, credentialsType: String = "default") extends AutoCloseable {

  lazy val client: AmazonS3 = {
    if (region == "" || region == null) {
      throw new InvalidArgumentException("Region argument is mandatory to create Amazon S3 client.")
    }
    var credentialParams = CredentialParams(accessKeyId, secretAccessKey, sessionToken, awsProfile, region)
    if (credentialsType == "public" || credentialsType == "community") {
      credentialParams = CredentialParams("anonymous", "", "", "", region)
    }
    val awsCredentials = new AWSTokenCredentials
    val credentials: Option[AWSCredentials] = awsCredentials.buildCredentials(credentialParams)

    getAmazonS3Client(credentials)
  }

  private def getAmazonS3Client(credentials: Option[AWSCredentials]): AmazonS3 = {
    val config = new ClientConfiguration()
    val timeout = ConfigLoader.getConfigIntValue(ConfigHelper.s3SocketTimeout)
    config.setSocketTimeout(timeout)

    val s3Client = {
      if (credentials.isDefined) {
        AmazonS3ClientBuilder.standard()
          .withCredentials(new AWSStaticCredentialsProvider(credentials.get))
          .withClientConfiguration(config)
      } else {
        println("Warning unable to build AWS credential via AWSGateway chain, some parameter is missing or malformed." +
          " S3 integration may not work well.")
        AmazonS3ClientBuilder.standard()
          .withClientConfiguration(config)
      }
    }

    s3Client.withRegion(region).build()
  }

  def getMetadata(s3Path: String, folder: String, bucket: String): List[ResourceMetadata] = {
    val metaFile = getS3File(s3Path, folder, "metadata.json")
    val obj = client.getObject(bucket, metaFile)
    val metadata = ResourceMetadata.readResources(obj.getObjectContent)
    metadata
  }

  def getS3File(parts: String*): String = {
    parts
      .map(part => part.stripSuffix("/"))
      .filter(part => part.nonEmpty)
      .mkString("/")
  }

  def doesS3ObjectExist(bucket: String, s3FilePath: String): Boolean = {
    try {
      client.getObjectMetadata(bucket, s3FilePath)
      true
    } catch {
      case e: AmazonServiceException => if (e.getStatusCode == 404) false else throw e
    }
  }

  def getS3Object(bucket: String, s3FilePath: String, tmpFile: File): ObjectMetadata = {
    val req = new GetObjectRequest(bucket, s3FilePath)
    client.getObject(req, tmpFile)
  }

  def getS3DownloadSize(s3Path: String, folder: String, fileName: String, bucket: String): Option[Long] = {
    try {
      val s3FilePath = getS3File(s3Path, folder, fileName)
      val meta = client.getObjectMetadata(bucket, s3FilePath)
      Some(meta.getContentLength)
    } catch {
      case e: AmazonServiceException => if (e.getStatusCode == 404) None else throw e
    }
  }

  def copyFileToS3(bucket: String, s3FilePath: String, sourceFilePath: String): PutObjectResult = {
    val sourceFile = new File("file://" + sourceFilePath)
    client.putObject(bucket, s3FilePath, sourceFile)
  }

  def copyInputStreamToS3(bucket: String, s3FilePath: String, sourceFilePath: String): PutObjectResult = {
    val fileSystem = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)
    val inputStream = fileSystem.open(new Path(sourceFilePath))
    client.putObject(bucket, s3FilePath, inputStream, new ObjectMetadata())
  }

  override def close(): Unit = {
    client.shutdown()
  }

}
