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

trait HasStorage extends HasStorageProperties {

  val storagePath = new ExternalResourceParam(this, "storagePath", "path to file")

  def setStoragePath(path: String, readAs: String): this.type = set(storagePath, new ExternalResource(path, readAs, Map.empty[String, String]))

  def getStoragePath: ExternalResource = $(storagePath)

  protected def index(storageSourcePath: String, connection: RocksDBConnection, resource: ExternalResource)

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

  private def copyIndexToCluster(localFile: String, dst: Path, spark: SparkContext): String = {
    val fs = new Path(localFile).getFileSystem(spark.hadoopConfiguration)
    val src = new Path(localFile)

    /** This fails if working on local file system, because spark.addFile will detect simoultaneous writes on same location and fail */
    fs.copyFromLocalFile(false, true, src, dst)
    fs.deleteOnExit(dst)

    spark.addFile(dst.toString, true)

    dst.toString
  }

  private def copyIndexToLocal(source: Path, destination: Path, context: SparkContext) = {
    /** if we don't do a copy, and just move, it will all fail when re-saving utilized storage because of bad crc */
    val fs = source.getFileSystem(context.hadoopConfiguration)
    fs.copyFromLocalFile(false, true, source, destination)
    fs.deleteOnExit(source)
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

    val clusterFileName: String = new Path(database).toString

    val destinationScheme = new Path(clusterFileName).getFileSystem(sparkContext.hadoopConfiguration).getScheme
    val fileSystem = FileSystem.get(sparkContext.hadoopConfiguration)

    val clusterTmpLocation = {
      ConfigHelper.getConfigValue(ConfigHelper.storageTmpDir).map(p => new Path(p)).getOrElse(
        sparkContext.hadoopConfiguration.get("hadoop.tmp.dir")
      )
    }
    val clusterFilePath = Path.mergePaths(new Path(fileSystem.getUri.toString + clusterTmpLocation), new Path("/"+clusterFileName))

    // 1 and 2.  Copy to local and Index Word Embeddings
    indexDatabase(sourceEmbeddingsPath, tmpLocalDestination.toString, sparkContext, resource)

    if (destinationScheme == "file") {
      copyIndexToLocal(new Path(tmpLocalDestination.toString), new Path(RocksDBConnection.getLocalPath(clusterFileName)), sparkContext)
    } else {
      // 2. Copy WordEmbeddings to cluster
      copyIndexToCluster(tmpLocalDestination.toString, clusterFilePath, sparkContext)
      FileHelper.delete(tmpLocalDestination.toString)
    }

    // 3. Create Spark Embeddings
    RocksDBConnection.getOrCreate(clusterFileName)
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

  def indexStorage(spark: SparkSession, resource: ExternalResource): Unit = {
    databases.foreach(database =>
      preload(
        resource,
        spark,
        database
      )
    )
  }

}
