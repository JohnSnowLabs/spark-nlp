package com.johnsnowlabs.nlp.embeddings

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID

import com.johnsnowlabs.util.{ConfigHelper, FileHelper}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.ivy.util.FileUtil
import org.apache.spark.{SparkContext, SparkFiles}
import org.slf4j.LoggerFactory

/*
  1. Copy Embeddings to local tmp file
  2. Index Embeddings if need
  3. Copy Index to cluster
  4. Open RocksDb based Embeddings on local index (lazy)
 */
class ClusterWordEmbeddings(val clusterFilePath: String, val dim: Int, val caseSensitive: Boolean) extends Serializable {

  @transient
  private var localRetriever: Option[WordEmbeddingsRetriever] = None

  def getOrCreateLocalRetriever: WordEmbeddingsRetriever = {

    /** Synchronized removed. Verify */
    if (localRetriever.isEmpty) {
      // Have to copy file because RockDB changes it and Spark rises Exception
      val src = SparkFiles.get(clusterFilePath)
      val workPath = src + "_work"

      if (!new File(workPath).exists()) {
        require(new File(src).exists(), s"indexed embeddings at $src not found")
        FileUtil.deepCopy(new File(src), new File(workPath), null, false)
      }

      localRetriever = Some(WordEmbeddingsRetriever(workPath, dim, caseSensitive))
    }

    localRetriever.get
  }
}

object ClusterWordEmbeddings {

  protected val logger = LoggerFactory.getLogger(classOf[ClusterWordEmbeddings])
  private def indexEmbeddings(sourceEmbeddingsPath: String,
                              localFile: String,
                              format: WordEmbeddingsFormat.Format,
                              spark: SparkContext): Unit = {

    val uri = new File(sourceEmbeddingsPath).toURI
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
    val uri = new File(localFile).toURI
    val fs = FileSystem.get(uri, spark.hadoopConfiguration)
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
            format: WordEmbeddingsFormat.Format): ClusterWordEmbeddings = {

    val localFile = {
      Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_idx")
        .toAbsolutePath
    }

    val clusterFilePath: String = {
      val name = localFile.toFile.getName
      Path.mergePaths(new Path("/embeddings"), new Path(name)).toString
    }

    // 1 and 2.  Copy to local and Index Word Embeddings
    indexEmbeddings(sourceEmbeddingsPath, localFile.toString, format, spark)

    // 2. Copy WordEmbeddings to cluster
    copyIndexToCluster(localFile.toString, clusterFilePath, spark)
    FileHelper.delete(localFile.toString)

    // 3. Create Spark Embeddings
    new ClusterWordEmbeddings(clusterFilePath, dim, caseSensitive)
  }
}
