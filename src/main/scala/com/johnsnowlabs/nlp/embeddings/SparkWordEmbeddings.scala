package com.johnsnowlabs.nlp.embeddings

import java.io.File
import java.nio.file.Files
import java.util.UUID

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.ivy.util.FileUtil
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkContext, SparkFiles}

/*
  1. Copy Embeddings to local tmp file
  2. Index Embeddings if need
  3. Copy Index to cluster
  4. Open RocksDb based Embeddings on local index (lazy)
 */
class SparkWordEmbeddings(val clusterFilePath: String, val dim: Int) extends Serializable {

  // Have to copy file because RockDB changes it and Spark rises Exception
  val src = SparkFiles.get(clusterFilePath)
  val workPath = src + "_work"

  @transient
  private var wordEmbeddingsValue: WordEmbeddings = null

  def wordEmbeddings: WordEmbeddings = {
    if (wordEmbeddingsValue == null) {
      if (!new File(workPath).exists()) {
        require(new File(src).exists(), s"file $src must be added to sparkContext")
        FileUtil.deepCopy(new File(src), new File(workPath), null, false)
      }

      wordEmbeddingsValue = WordEmbeddings(workPath, dim)
    }

    wordEmbeddingsValue
  }
}

object SparkWordEmbeddings {

  private def indexEmbeddings(sourceEmbeddingsPath: String,
                              localFile: String,
                              format: WordEmbeddingsFormat.Format,
                              spark: SparkContext): Unit = {

    val fs = FileSystem.get(spark.hadoopConfiguration)

    if (format == WordEmbeddingsFormat.TEXT) {

      val tmpFile = Files.createTempFile("embeddings", ".txt").toAbsolutePath
      fs.copyToLocalFile(new Path(sourceEmbeddingsPath), new Path(tmpFile.toString))
      WordEmbeddingsIndexer.indexText(tmpFile.toString, localFile)
      Files.delete(tmpFile)
    }
    else if (format == WordEmbeddingsFormat.BINARY) {

      val tmpFile = Files.createTempFile("embeddings", ".bin").toAbsolutePath
      fs.copyToLocalFile(new Path(sourceEmbeddingsPath), new Path(tmpFile.toString))
      WordEmbeddingsIndexer.indexBinary(tmpFile.toString, localFile)
      Files.delete(tmpFile)
    }
    else if (format == WordEmbeddingsFormat.SPARKNLP) {
      
      val hdfs = FileSystem.get(spark.hadoopConfiguration)
      hdfs.copyToLocalFile(new Path(sourceEmbeddingsPath), new Path(localFile))
    }
  }

  private def copyIndexToCluster(localFile: String, clusterFilePath: String, spark: SparkContext): String = {
    val fs = FileSystem.get(spark.hadoopConfiguration)

    val src = new Path(localFile)
    val dst = Path.mergePaths(fs.getHomeDirectory, new Path(clusterFilePath))

    fs.copyFromLocalFile(false, true, src, dst)
    fs.deleteOnExit(dst)

    spark.addFile(dst.toString, true)

    dst.toString
  }


  def apply(spark: SparkContext,
            sourceEmbeddingsPath: String,
            dim: Int,
            format: WordEmbeddingsFormat.Format): SparkWordEmbeddings = {

    val localFile: String = {
      Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_idx")
        .toAbsolutePath.toString
    }

    val clusterFilePath: String = {
      val name = new File(localFile).getName
      Path.mergePaths(new Path("/embeddings"), new Path(name)).toString
    }

    // 1 and 2.  Copy to local and Index Word Embeddings
    indexEmbeddings(sourceEmbeddingsPath, localFile, format, spark)

    // 2. Copy WordEmbeddings to cluster
    copyIndexToCluster(localFile, clusterFilePath, spark)

    // 3. Create Spark Embeddings
    new SparkWordEmbeddings(clusterFilePath, dim)
  }
}
