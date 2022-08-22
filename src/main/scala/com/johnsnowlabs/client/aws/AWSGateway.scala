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

package com.johnsnowlabs.client.aws

import com.amazonaws.auth.{AWSCredentials, AWSStaticCredentialsProvider}
import com.amazonaws.services.pi.model.InvalidArgumentException
import com.amazonaws.services.s3.model.{
  GetObjectRequest,
  ObjectMetadata,
  PutObjectResult,
  S3Object
}
import com.amazonaws.services.s3.transfer.{Transfer, TransferManagerBuilder}
import com.amazonaws.services.s3.{AmazonS3, AmazonS3ClientBuilder}
import com.amazonaws.{AmazonClientException, AmazonServiceException, ClientConfiguration}
import com.johnsnowlabs.client.CredentialParams
import com.johnsnowlabs.nlp.pretrained.ResourceMetadata
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.slf4j.{Logger, LoggerFactory}

import java.io.File

class AWSGateway(
    accessKeyId: String = ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalAccessKeyId),
    secretAccessKey: String =
      ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalSecretAccessKey),
    sessionToken: String =
      ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalSessionToken),
    awsProfile: String = ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalProfileName),
    region: String = ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalRegion),
    credentialsType: String = "default")
    extends AutoCloseable {

  protected val logger: Logger = LoggerFactory.getLogger(this.getClass.toString)

  lazy val client: AmazonS3 = {
    if (region.isEmpty || region == null) {
      throw new InvalidArgumentException(
        "Region argument is mandatory to create Amazon S3 client.")
    }
    var credentialParams =
      CredentialParams(accessKeyId, secretAccessKey, sessionToken, awsProfile, region)
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
        AmazonS3ClientBuilder
          .standard()
          .withCredentials(new AWSStaticCredentialsProvider(credentials.get))
          .withClientConfiguration(config)
      } else {
        val warning_message =
          "Unable to build AWS credential via AWSGateway chain, some parameter is missing or" +
            " malformed. S3 integration may not work well."
        logger.warn(warning_message)
        AmazonS3ClientBuilder
          .standard()
          .withClientConfiguration(config)
      }
    }

    s3Client.withRegion(region).build()
  }

  def getMetadata(s3Path: String, folder: String, bucket: String): List[ResourceMetadata] = {
    val metaFile = getS3File(s3Path, folder, "metadata.json")
    val obj = getObjectFromS3(bucket, metaFile)
    val metadata = ResourceMetadata.readResources(obj.getObjectContent)
    metadata
  }

  private def getObjectFromS3(bucket: String, key: String): S3Object = {
    try {
      this.client.getObject(bucket, key)
    } catch {
      case _: AmazonClientException =>
        val anonymousCredentialParams = CredentialParams("anonymous", "", "", "", region)
        val awsCredentials = new AWSAnonymousCredentials
        val credentials: Option[AWSCredentials] =
          awsCredentials.buildCredentials(anonymousCredentialParams)
        val client = getAmazonS3Client(credentials)
        client.getObject(bucket, key)
    }
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

  def getS3DownloadSize(
      s3Path: String,
      folder: String,
      fileName: String,
      bucket: String): Option[Long] = {
    try {
      val s3FilePath = getS3File(s3Path, folder, fileName)
      val meta = client.getObjectMetadata(bucket, s3FilePath)
      Some(meta.getContentLength)
    } catch {
      case e: AmazonServiceException => if (e.getStatusCode == 404) None else throw e
    }
  }

  def copyFileToS3(
      bucket: String,
      s3FilePath: String,
      sourceFilePath: String): PutObjectResult = {
    val sourceFile = new File("file://" + sourceFilePath)
    client.putObject(bucket, s3FilePath, sourceFile)
  }

  def copyInputStreamToS3(
      bucket: String,
      s3FilePath: String,
      sourceFilePath: String): PutObjectResult = {
    val fileSystem = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)
    val inputStream = fileSystem.open(new Path(sourceFilePath))
    client.putObject(bucket, s3FilePath, inputStream, new ObjectMetadata())
  }

  def downloadFilesFromDirectory(
      bucketName: String,
      keyPrefix: String,
      directoryPath: File): Unit = {
    val transferManager = TransferManagerBuilder
      .standard()
      .withS3Client(client)
      .build()
    try {
      val multipleFileDownload =
        transferManager.downloadDirectory(bucketName, keyPrefix, directoryPath)
      println(multipleFileDownload.getDescription)
      waitForCompletion(multipleFileDownload)
    } catch {
      case e: AmazonServiceException =>
        throw new AmazonServiceException(
          "Amazon service error when downloading files from S3 directory: " + e.getMessage)
    }
    transferManager.shutdownNow()
  }

  private def waitForCompletion(transfer: Transfer): Unit = {
    try transfer.waitForCompletion()
    catch {
      case e: AmazonServiceException =>
        throw new AmazonServiceException("Amazon service error: " + e.getMessage)
      case e: AmazonClientException =>
        throw new AmazonClientException("Amazon client error: " + e.getMessage)
      case e: InterruptedException =>
        throw new InterruptedException("Transfer interrupted: " + e.getMessage)
    }
  }

  override def close(): Unit = {
    client.shutdown()
  }

}
