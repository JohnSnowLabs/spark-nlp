package com.johnsnowlabs.storage

import java.nio.file.{Files, Paths, StandardCopyOption}
import java.util.UUID

import com.amazonaws.auth.DefaultAWSCredentialsProviderChain
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.util.{ConfigHelper, FileHelper}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

trait HasStorage extends HasStorageRef {

  val storagePath = new ExternalResourceParam(this, "storagePath", "path to file")

  def setStoragePath(path: String, readAs: String): this.type = set(storagePath, new ExternalResource(path, readAs, Map.empty[String, String]))

  def getStoragePath: ExternalResource = $(storagePath)

  protected val missingRefMsg: String = s"Please set storageRef param in $this."

  protected def index(storageSourcePath: String, connection: RocksDBConnection, resource: ExternalResource): Unit

  private def indexDatabase(storageSourcePath: String,
                            localFile: String,
                            spark: SparkContext,
                            resource: ExternalResource
                           ): Unit = {

    val uri = new java.net.URI(storageSourcePath.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.hadoopConfiguration)

    lazy val connection = RocksDBConnection.getOrCreate(localFile)

    if (new Path(storageSourcePath).getFileSystem(spark.hadoopConfiguration).getScheme != "file") {
      val tmpFile = Files.createTempFile("sparknlp_", ".str").toAbsolutePath.toString
      fs.copyToLocalFile(new Path(storageSourcePath), new Path(tmpFile))
      index(tmpFile, connection, resource)
      FileHelper.delete(tmpFile)
    } else {
      index(storageSourcePath, connection, resource)
    }

  }

  private def preload(
                       resource: ExternalResource,
                       spark: SparkSession,
                       database: String): RocksDBConnection = {

    val sourceEmbeddingsPath = importIfS3(resource.path, spark).toUri.toString
    val sparkContext = spark.sparkContext

    val tmpLocalDestination = {
      Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_idx")
        .toAbsolutePath
    }

    val fileSystem = FileSystem.get(sparkContext.hadoopConfiguration)

    val locator = StorageLocator(database, spark, fileSystem)

    // 1 and 2.  Copy to local and Index Word Embeddings
    indexDatabase(sourceEmbeddingsPath, tmpLocalDestination.toString, sparkContext, resource)

    StorageHelper.sendToCluster(tmpLocalDestination.toString, locator.clusterFilePath, locator.clusterFileName, locator.destinationScheme, sparkContext)

    // 3. Create Spark Embeddings
    RocksDBConnection.getOrCreate(locator.clusterFileName)
  }


  private def importIfS3(path: String, spark: SparkSession): Path = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    var src = new Path(path)
    //if the path contains s3a download to local cache if not present
    if (uri.getScheme != null) {
      if (uri.getScheme.equals("s3a")) {
        var accessKeyId = ConfigHelper.getConfigValue(ConfigHelper.accessKeyId)
        var secretAccessKey = ConfigHelper.getConfigValue(ConfigHelper.secretAccessKey)
        if (accessKeyId.isEmpty || secretAccessKey.isEmpty) {
          val defaultCred = new DefaultAWSCredentialsProviderChain().getCredentials
          accessKeyId = Some(defaultCred.getAWSAccessKeyId)
          secretAccessKey = Some(defaultCred.getAWSSecretKey)
        }
        var old_key = ""
        var old_secret = ""
        if (spark.sparkContext.hadoopConfiguration.get("fs.s3a.access.key") != null) {
          old_key = spark.sparkContext.hadoopConfiguration.get("fs.s3a.access.key")
          old_secret = spark.sparkContext.hadoopConfiguration.get("fs.s3a.secret.key")
        }
        try {
          val dst = new Path(ResourceDownloader.cacheFolder, src.getName)
          if (!Files.exists(Paths.get(dst.toUri.getPath))) {
            //download s3 resource locally using config keys
            spark.sparkContext.hadoopConfiguration.set("fs.s3a.access.key", accessKeyId.get)
            spark.sparkContext.hadoopConfiguration.set("fs.s3a.secret.key", secretAccessKey.get)
            val s3fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

            val dst_tmp = new Path(ResourceDownloader.cacheFolder, src.getName + "_tmp")


            s3fs.copyToLocalFile(src, dst_tmp)
            // rename to original file
            val path = Files.move(
              Paths.get(dst_tmp.toUri.getRawPath),
              Paths.get(dst.toUri.getRawPath),
              StandardCopyOption.REPLACE_EXISTING
            )

          }
          src = new Path(dst.toUri.getPath)
        }
        finally {
          //reset the keys
          if (!old_key.equals("")) {
            spark.sparkContext.hadoopConfiguration.set("fs.s3a.access.key", old_key)
            spark.sparkContext.hadoopConfiguration.set("fs.s3a.secret.key", old_secret)
          }
        }

      }
    }
    src
  }

  protected def indexStorage(spark: SparkSession, resource: ExternalResource): Unit = {
    require(isDefined(storageRef), missingRefMsg)
    databases.foreach(database =>
      preload(
        resource,
        spark,
        database.toString
      )
    )
  }

}
