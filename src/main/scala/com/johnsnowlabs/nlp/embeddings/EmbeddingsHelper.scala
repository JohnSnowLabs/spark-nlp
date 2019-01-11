package com.johnsnowlabs.nlp.embeddings

import java.io.FileNotFoundException

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
    //if the path contains s3a setup the aws credentials from config if not already present
    if(path.startsWith("s3a")){
      if(spark.sparkContext.hadoopConfiguration.get("fs.s3a.access.key")==null) {
        val accessKeyId = ConfigHelper.getConfigValue(ConfigHelper.accessKeyId)
        val secretAccessKey = ConfigHelper.getConfigValue(ConfigHelper.secretAccessKey)


        if (accessKeyId.isEmpty || secretAccessKey.isEmpty)
          throw new SecurityException("AWS credentials not set in config")
        else {

          spark.sparkContext.hadoopConfiguration.set("fs.s3a.access.key", accessKeyId.get)
          spark.sparkContext.hadoopConfiguration.set("fs.s3a.secret.key", secretAccessKey.get)
        }

      }
    }
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    val src = new Path(path)

    if (fs.exists(src)) {
      ClusterWordEmbeddings(
        spark.sparkContext,
        src.toUri.toString,
        nDims,
        caseSensitiveEmbeddings,
        format,
        embeddingsRef
      )
    } else {
      throw new FileNotFoundException(s"embeddings not found in $path")
    }

  }

  def load(
            indexPath: String,
            nDims: Int,
            caseSensitive: Boolean
          ): ClusterWordEmbeddings = {
    new ClusterWordEmbeddings(indexPath, nDims, caseSensitive)
  }

  def getFromAnnotator(annotator: ModelWithWordEmbeddings): ClusterWordEmbeddings = {
    annotator.getClusterEmbeddings
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
