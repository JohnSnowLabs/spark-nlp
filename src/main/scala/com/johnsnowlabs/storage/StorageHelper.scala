package com.johnsnowlabs.storage

import java.nio.file.{Files, Paths, StandardCopyOption}
import java.util.UUID

import com.amazonaws.auth.DefaultAWSCredentialsProviderChain
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.util.{ConfigHelper, FileHelper}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.{SparkContext, SparkFiles}
import org.apache.spark.sql.SparkSession


trait StorageHelper[A, +B <: RocksDBReader[A]] extends Serializable {

  val StorageFormats: Enumeration

  val filesPrefix: String

  protected def createConnection(filename: String, caseSensitive: Boolean): B

  protected def indexStorage(storageSourcePath: String,
                             localFile: String,
                             format: StorageFormats.Value,
                             spark: SparkContext
                            ): Unit

  def load(
            path: String,
            spark: SparkSession,
            format: String,
            caseSensitiveEmbeddings: Boolean,
            embeddingsRef: String
          ): RocksDBReader[A] = {
    load(
      path,
      spark,
      StorageFormats.withName(format.toUpperCase),
      caseSensitiveEmbeddings,
      embeddingsRef
    )
  }

  def load(
            path: String,
            spark: SparkSession,
            format: StorageFormats.Value,
            caseSensitiveEmbeddings: Boolean,
            embeddingsRef: String): RocksDBReader[A] = {

    val src = importIfS3(path, spark)
    apply(
      spark.sparkContext,
      src.toUri.toString,
      caseSensitiveEmbeddings,
      format,
      embeddingsRef
    )
  }

  def getClusterFilename(embeddingsRef: String): String = {
    Path.mergePaths(new Path(filesPrefix), new Path(embeddingsRef)).toString
  }

  def apply(spark: SparkContext,
            sourceEmbeddingsPath: String,
            caseSensitive: Boolean,
            format: StorageFormats.Value,
            storageRef: String): RocksDBReader[A] = {

    val tmpLocalDestination = {
      Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_idx")
        .toAbsolutePath
    }

    val clusterFileName: String = getClusterFilename(storageRef)

    val destinationScheme = new Path(clusterFileName).getFileSystem(spark.hadoopConfiguration).getScheme
    val fileSystem = FileSystem.get(spark.hadoopConfiguration)

    val clusterTmpLocation = {
      ConfigHelper.getConfigValue(ConfigHelper.storageTmpDir).map(new Path(_)).getOrElse(
        spark.hadoopConfiguration.get("hadoop.tmp.dir")
      )
    }
    val clusterFilePath = Path.mergePaths(new Path(fileSystem.getUri.toString + clusterTmpLocation), new Path("/"+clusterFileName))

    // 1 and 2.  Copy to local and Index Word Embeddings
    indexStorage(sourceEmbeddingsPath, tmpLocalDestination.toString, format, spark)

    if (destinationScheme == "file") {
      StorageHelper.copyIndexToLocal(new Path(tmpLocalDestination.toString), new Path(StorageHelper.getLocalPath(clusterFileName)), spark)
    } else {
      // 2. Copy WordEmbeddings to cluster
      StorageHelper.copyIndexToCluster(tmpLocalDestination.toString, clusterFilePath, spark)
      FileHelper.delete(tmpLocalDestination.toString)
    }

    // 3. Create Spark Embeddings
    createConnection(clusterFileName, caseSensitive)
  }

  def load(
            indexPath: String,
            caseSensitive: Boolean
          ): RocksDBReader[A] = {
    createConnection(indexPath, caseSensitive)
  }

  def save(path: String, connection: RocksDBReader[A], spark: SparkSession): Unit = {
    StorageHelper.save(path, spark, connection.fileName.toString)
  }

  def importIfS3(path: String, spark: SparkSession): Path = {
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

}

object StorageHelper {

  def getLocalPath(fileName: String): String = {
    Path.mergePaths(new Path(SparkFiles.getRootDirectory()), new Path("/"+fileName)).toString
  }

  protected def save(path: String, spark: SparkSession, fileName: String): Unit = {
    val index = new Path(getLocalPath(fileName))

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dst = new Path(path)

    save(fs, index, dst)
  }

  def save(fs: FileSystem, index: Path, dst: Path): Unit = {
    fs.copyFromLocalFile(false, true, index, dst)
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


}

