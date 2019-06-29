package com.johnsnowlabs.nlp.embeddings

import java.nio.file.{Files, Paths, StandardCopyOption}

import com.amazonaws.auth.DefaultAWSCredentialsProviderChain
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.util.ConfigHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkFiles
import org.apache.spark.sql.SparkSession

object EmbeddingsHelper {

  def load(
            path: String,
            spark: SparkSession,
            format: String,
            embeddingsRef: String,
            nDims: Int,
            caseSensitiveEmbeddings: Boolean
          ): ClusterWordEmbeddings = {
    import WordEmbeddingsFormat._
    load(
      path,
      spark,
      str2frm(format),
      nDims,
      caseSensitiveEmbeddings,
      embeddingsRef
    )
  }

  def load(
            path: String,
            spark: SparkSession,
            format: WordEmbeddingsFormat.Format,
            nDims: Int,
            caseSensitiveEmbeddings: Boolean,
            embeddingsRef: String): ClusterWordEmbeddings = {

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
    ClusterWordEmbeddings(
      spark.sparkContext,
      src.toUri.toString,
      nDims,
      caseSensitiveEmbeddings,
      format,
      embeddingsRef
    )
  }

  def load(
            indexPath: String,
            nDims: Int,
            caseSensitive: Boolean
          ): ClusterWordEmbeddings = {
    new ClusterWordEmbeddings(indexPath, nDims, caseSensitive)
  }

  def getClusterFilename(embeddingsRef: String): String = {
    Path.mergePaths(new Path("/embd_"), new Path(embeddingsRef)).toString
  }

  def getLocalEmbeddingsPath(fileName: String): String = {
    Path.mergePaths(new Path(SparkFiles.getRootDirectory()), new Path(fileName)).toString
  }

  protected def save(path: String, spark: SparkSession, fileName: String): Unit = {
    val index = new Path(EmbeddingsHelper.getLocalEmbeddingsPath(fileName))

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dst = new Path(path)

    save(fs, index, dst)
  }

  def save(path: String, embeddings: ClusterWordEmbeddings, spark: SparkSession): Unit = {
    EmbeddingsHelper.save(path, spark, embeddings.fileName.toString)
  }

  def save(fs: FileSystem, index: Path, dst: Path): Unit = {
    fs.copyFromLocalFile(false, true, index, dst)
  }


}
