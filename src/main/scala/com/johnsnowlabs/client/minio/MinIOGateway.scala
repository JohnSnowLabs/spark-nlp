package com.johnsnowlabs.client.minio

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import io.minio.{DownloadObjectArgs, ListObjectsArgs, MinioClient, PutObjectArgs}
import org.apache.hadoop.fs.{FileSystem, Path}

class MinIOGateway(
    accessKeyId: String = ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalAccessKeyId),
    secretAccessKey: String =
      ConfigLoader.getConfigStringValue(ConfigHelper.awsExternalSecretAccessKey),
    endpoint: String =
      ConfigLoader.getConfigStringValue(ConfigHelper.externalClusterStorageURI)) {

  lazy val client: MinioClient = {
    import io.minio.MinioClient
    MinioClient.builder
      .endpoint(endpoint)
      .credentials(accessKeyId, secretAccessKey)
      .build
  }

  def copyFileToMinIO(bucket: String, sourceFilePath: String): Unit = {

    val fileName = sourceFilePath.split("/").last
    val minIOBucket = bucket.replace("s3://", "").split("/").head
    val bucketPath = bucket.substring(s"s3://$minIOBucket".length) + "/"
    val bucketFilePath = s"$bucketPath$fileName"

    val fileSystem = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)
    val inputStream = fileSystem.open(new Path(sourceFilePath))

    val objectStream = PutObjectArgs
      .builder()
      .bucket(minIOBucket)
      .`object`(bucketFilePath)
      .stream(inputStream, inputStream.available(), -1)
      .build()

    client.putObject(objectStream)
  }

  def downloadFilesFromDirectory(bucket: String, destinationPath: String): Unit = {

    val bucketParts = bucket.replace("s3://", "").split("/")
    val minIOBucket = bucketParts.head
    val bucketPrefix = bucket.substring(s"s3://$minIOBucket".length + 1)

    val listObjectsArgs = ListObjectsArgs
      .builder()
      .bucket(minIOBucket)
    if (bucketPrefix != "") listObjectsArgs.prefix(bucketPrefix)
    listObjectsArgs.recursive(true)

    val files = client.listObjects(listObjectsArgs.build())

    files.forEach { file =>
      val item = file.get()
      val objectPath = item.objectName()
      val localFilePath = s"$destinationPath/${objectPath.split("/").last}"

      client.downloadObject(
        DownloadObjectArgs
          .builder()
          .bucket(minIOBucket)
          .`object`(objectPath)
          .filename(localFilePath)
          .build())
    }

  }

}
