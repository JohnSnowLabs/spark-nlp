/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.client

import com.amazonaws.auth.profile.ProfileCredentialsProvider
import com.amazonaws.auth.{AWSCredentials, AnonymousAWSCredentials, BasicAWSCredentials, DefaultAWSCredentialsProviderChain}
import com.amazonaws.regions.RegionUtils
import com.amazonaws.services.s3.AmazonS3Client
import com.amazonaws.services.s3.model.{GetObjectRequest, ObjectMetadata, PutObjectResult}
import com.amazonaws.{AmazonClientException, AmazonServiceException, ClientConfiguration}
import com.johnsnowlabs.nlp.pretrained.ResourceMetadata
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import org.apache.hadoop.fs.{FileSystem, Path}

import java.io.File

class AWSGateway(accessKeyId: String, secretAccessKey: String, awsProfile: String, region: String,
                 credentialsType: String = "default")
  extends AutoCloseable {

  lazy val credentials: Option[AWSCredentials] = {
    credentialsType match {
      case "default" =>
        if (ConfigLoader.hasAwsCredentials) {
          buildAwsCredentials()
        } else {
          fetchCredentials()
        }
      case "proprietary" => if (ConfigLoader.hasFullAwsCredentials) buildAwsCredentials() else None
      case _ => Some(new AnonymousAWSCredentials())
    }
  }

  /* ToDo AmazonS3Client has been deprecated*/
  private lazy val client: AmazonS3Client = getAmazonS3Client(credentials)

  def buildAwsCredentials(): Option[AWSCredentials] = {
    if (awsProfile != "") {
      return Some(new ProfileCredentialsProvider(awsProfile).getCredentials)
    }
    if (accessKeyId == "" || secretAccessKey == "") {
      fetchCredentials()
    }
    else
      Some(new BasicAWSCredentials(accessKeyId, secretAccessKey))
  }

  def fetchCredentials(): Option[AWSCredentials] = {
    try {
      //check if default profile name works if not try
      Some(new ProfileCredentialsProvider("spark_nlp").getCredentials)
    } catch {
      case _: Exception =>
        try {

          Some(new DefaultAWSCredentialsProviderChain().getCredentials)
        } catch {
          case _: AmazonClientException =>
            if (ResourceHelper.spark.sparkContext.hadoopConfiguration.get("fs.s3a.access.key") != null) {

              val key = ResourceHelper.spark.sparkContext.hadoopConfiguration.get("fs.s3a.access.key")
              val secret = ResourceHelper.spark.sparkContext.hadoopConfiguration.get("fs.s3a.secret.key")

              Some(new BasicAWSCredentials(key, secret))
            } else {
              Some(new AnonymousAWSCredentials())
            }
          case e: Exception => throw e

        }
    }

  }

  def getAmazonS3Client(credentials: Option[AWSCredentials]): AmazonS3Client = {
    val regionObj = RegionUtils.getRegion(region)

    val config = new ClientConfiguration()
    val timeout = ConfigLoader.getConfigIntValue(ConfigHelper.s3SocketTimeout)
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
    val result = new Path(sourceFilePath)
    println(s"In copyFileToS3 result: ${result.toUri.getRawPath}")
    val sourceFile = new File("file://" + sourceFilePath)
    client.putObject(bucket, s3FilePath, sourceFile)
  }

  def copyInputStreamToS3(bucket: String, s3FilePath: String, sourceFilePath: String) = {
    val fileSystem = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)
    val inputStream = fileSystem.open(new Path(sourceFilePath))
    client.putObject(bucket, s3FilePath, inputStream, new ObjectMetadata())
  }

  override def close(): Unit = {
    client.shutdown()
  }

}
