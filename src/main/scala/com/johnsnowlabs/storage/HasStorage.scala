package com.johnsnowlabs.storage

import java.nio.file.{Files, Paths, StandardCopyOption}
import java.util.UUID

import com.amazonaws.auth.DefaultAWSCredentialsProviderChain
import com.johnsnowlabs.nlp.HasCaseSensitiveProperties
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.storage.Database.Name
import com.johnsnowlabs.util.{ConfigHelper, FileHelper}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

trait HasStorage extends HasStorageRef with HasCaseSensitiveProperties {

  val storagePath = new ExternalResourceParam(this, "storagePath", "path to file")

  def setStoragePath(path: String, readAs: String): this.type = set(storagePath, new ExternalResource(path, readAs, Map.empty[String, String]))

  def getStoragePath: ExternalResource = $(storagePath)

  protected val missingRefMsg: String = s"Please set storageRef param in $this."

  protected def index(
                       storageSourcePath: String,
                       readAs: ReadAs.Value,
                       writers: Map[Database.Name, StorageWriter[_]],
                       readOptions: Map[String, String] = Map()
                     ): Unit

  protected def createWriter(database: Name, connection: RocksDBConnection): StorageWriter[_]

  private def indexDatabases(
                              databases: Array[Database.Value],
                              storageSourcePath: String,
                              localFiles: Array[String],
                              spark: SparkContext,
                              resource: ExternalResource
                           ): Unit = {

    require(databases.length == localFiles.length, "Storage temp locations must be equal to the amount of databases")

    val uri = new java.net.URI(storageSourcePath.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.hadoopConfiguration)

    lazy val connections = databases.zip(localFiles)
      .map{ case (database, localFile) => (database, RocksDBConnection.getOrCreate(localFile))}

    val writers = connections.map{ case (db, conn) =>
      (db, createWriter(db, conn))
    }.toMap[Database.Name, StorageWriter[_]]

    if (new Path(storageSourcePath).getFileSystem(spark.hadoopConfiguration).getScheme != "file") {
      /** ToDo: What if the file is too large to copy to local? Index directly from hadoop? */
      val tmpFile = Files.createTempFile("sparknlp_", ".str").toAbsolutePath.toString
      fs.copyToLocalFile(new Path(storageSourcePath), new Path(tmpFile))
      index(tmpFile, resource.readAs, writers, resource.options)
      FileHelper.delete(tmpFile)
    } else {
      index(storageSourcePath, resource.readAs, writers, resource.options)
    }

    writers.values.foreach(_.close())
    connections.map(_._2).foreach(_.close())
  }

  private def preload(
                       resource: ExternalResource,
                       spark: SparkSession,
                       databases: Array[Database.Value]
                     ): Array[RocksDBConnection] = {

    val sourceEmbeddingsPath = importIfS3(resource.path, spark).toUri.toString
    val sparkContext = spark.sparkContext

    val tmpLocalDestinations = {
      databases.map( _ =>
        Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_idx")
          .toAbsolutePath.toString
      )
    }

    val fileSystem = FileSystem.get(sparkContext.hadoopConfiguration)

    indexDatabases(databases, sourceEmbeddingsPath, tmpLocalDestinations, sparkContext, resource)

    val locators = databases.map(database => StorageLocator(database.toString, $(storageRef), spark, fileSystem))

    tmpLocalDestinations.zip(locators).foreach{case (tmpLocalDestination, locator) => {
      StorageHelper.sendToCluster(tmpLocalDestination.toString, locator.clusterFilePath, locator.clusterFileName, locator.destinationScheme, sparkContext)
    }}

    // 3. Create Spark Embeddings
    locators.map(locator => RocksDBConnection.getOrCreate(locator.clusterFileName))
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
    preload(
      resource,
      spark,
      databases
    )
  }

}
