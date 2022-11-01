package com.johnsnowlabs.client.minio

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{ConfigHelper, ConfigLoader}
import io.minio.{MinioClient, PutObjectArgs}
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

  def copyResourceToMinIO(fileName: String, bucket: String, sourceFilePath: String): Unit = {

    val fileSystem = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)
    val inputStream = fileSystem.open(new Path(sourceFilePath))

    val objectStream = PutObjectArgs
      .builder()
      .bucket(bucket)
      .`object`(fileName)
      .stream(inputStream, inputStream.available(), -1)
      .build()

    client.putObject(objectStream)
  }

}
