package com.johnsnowlabs.nlp.embeddings

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID

import com.johnsnowlabs.util.{ConfigHelper, FileHelper}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.ivy.util.FileUtil
import org.apache.spark.{SparkContext, SparkFiles}

/*
  1. Copy Embeddings to local tmp file
  2. Index Embeddings if need
  3. Copy Index to cluster
  4. Open RocksDb based Embeddings on local index (lazy)
 */
class ClusterWordEmbeddings(val clusterFilePath: String, val dim: Int, val caseSensitive: Boolean) extends Serializable {

  def getLocalRetriever: WordEmbeddingsRetriever = {

    /** Synchronized removed. Verify */
    // Have to copy file because RockDB changes it and Spark rises Exception
    val src = SparkFiles.get(clusterFilePath)
    val workPath = src + "_work"

    if (!new File(workPath).exists()) {
      require(new File(src).exists(), s"Indexed embeddings at $src not found or not included. Call EmbeddingsHelper.load()")
      FileUtil.deepCopy(new File(src), new File(workPath), null, true)
    }

    WordEmbeddingsRetriever(workPath, dim, caseSensitive)
  }
}

object ClusterWordEmbeddings {

  private def indexEmbeddings(sourceEmbeddingsPath: String,
                              localFile: String,
                              format: WordEmbeddingsFormat.Format,
                              spark: SparkContext): Unit = {

    val uri = new java.net.URI(sourceEmbeddingsPath.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.hadoopConfiguration)

    if (format == WordEmbeddingsFormat.TEXT) {

      val tmpFile = Files.createTempFile("embeddings", ".txt").toAbsolutePath.toString
      fs.copyToLocalFile(new Path(sourceEmbeddingsPath), new Path(tmpFile))
      WordEmbeddingsIndexer.indexText(tmpFile, localFile)
      FileHelper.delete(tmpFile)
    }
    else if (format == WordEmbeddingsFormat.BINARY) {

      val tmpFile = Files.createTempFile("embeddings", ".bin").toAbsolutePath.toString
      fs.copyToLocalFile(new Path(sourceEmbeddingsPath), new Path(tmpFile))
      WordEmbeddingsIndexer.indexBinary(tmpFile, localFile)
      FileHelper.delete(tmpFile)
    }
    else if (format == WordEmbeddingsFormat.SPARKNLP) {

      fs.copyToLocalFile(new Path(sourceEmbeddingsPath), new Path(localFile))
      val fileName = new Path(sourceEmbeddingsPath).getName

      FileUtil.deepCopy(Paths.get(localFile, fileName).toFile, Paths.get(localFile).toFile, null, true)
      FileHelper.delete(Paths.get(localFile, fileName).toString)
    }
  }

  private def copyIndexToCluster(localFile: String, clusterFilePath: String, spark: SparkContext): String = {
    val fs = new Path(localFile).getFileSystem(spark.hadoopConfiguration)
    val cfs = FileSystem.get(spark.hadoopConfiguration)
    val src = new Path(localFile)
    val clusterTmpLocation = {
      ConfigHelper.getConfigValue(ConfigHelper.embeddingsTmpDir).map(new Path(_)).getOrElse(
        new Path(cfs.getScheme, "", spark.hadoopConfiguration.get("hadoop.tmp.dir"))
      )
    }
    val dst = Path.mergePaths(clusterTmpLocation, new Path(clusterFilePath))

    fs.copyFromLocalFile(false, true, src, dst)
    fs.deleteOnExit(dst)

    spark.addFile(dst.toString, true)

    dst.toString
  }


  def apply(spark: SparkContext,
            sourceEmbeddingsPath: String,
            dim: Int,
            caseSensitive: Boolean,
            format: WordEmbeddingsFormat.Format,
            embeddingsRef: String): ClusterWordEmbeddings = {

    val localDestination = {
      Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_idx")
        .toAbsolutePath
    }

    val clusterFilePath: String = {
      EmbeddingsHelper.getClusterPath(embeddingsRef)
    }

    // 1 and 2.  Copy to local and Index Word Embeddings
    indexEmbeddings(sourceEmbeddingsPath, localDestination.toString, format, spark)

    // 2. Copy WordEmbeddings to cluster
    copyIndexToCluster(localDestination.toString, clusterFilePath, spark)
    FileHelper.delete(localDestination.toString)

    // 3. Create Spark Embeddings
    new ClusterWordEmbeddings(embeddingsRef, dim, caseSensitive)
  }
}
