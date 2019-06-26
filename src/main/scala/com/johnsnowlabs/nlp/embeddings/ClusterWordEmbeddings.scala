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
class ClusterWordEmbeddings(val fileName: String, val dim: Int, val caseSensitive: Boolean) extends Serializable {

  @transient private var embds: WordEmbeddingsRetriever = null

  def getLocalRetriever: WordEmbeddingsRetriever = {
    val localPath = EmbeddingsHelper.getLocalEmbeddingsPath(fileName)
    if (Option(embds).isDefined)
      embds
    else if (new File(localPath).exists()) {
      embds = WordEmbeddingsRetriever(localPath, dim, caseSensitive)
      embds
    }
    else {
      val localFromClusterPath = SparkFiles.get(fileName)
      require(new File(localFromClusterPath).exists(), s"Embeddings not found under given ref ${fileName.replaceAll("/embd_", "")}\n" +
        s" This usually means:\n\n1. You have not loaded any embeddings under such embeddings ref\n2." +
        s" You are trying to use cluster mode without a proper shared filesystem.\n3. source was not provided to WordEmbeddings" +
        s"\n4. If you are trying to reutilize previous embeddings, make sure you use such ref here. ")
      embds = WordEmbeddingsRetriever(localFromClusterPath, dim, caseSensitive)
      embds
    }
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

  private def copyIndexToCluster(localFile: String, dst: Path, spark: SparkContext): String = {
    val fs = new Path(localFile).getFileSystem(spark.hadoopConfiguration)
    val src = new Path(localFile)

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

    val tmpLocalDestination = {
      Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_idx")
        .toAbsolutePath
    }

    val clusterFileName: String = {
      EmbeddingsHelper.getClusterFilename(embeddingsRef)
    }

    val destinationScheme = new Path(clusterFileName).getFileSystem(spark.hadoopConfiguration).getScheme
    val fileSystem = FileSystem.get(spark.hadoopConfiguration)

    val clusterTmpLocation = {
      ConfigHelper.getConfigValue(ConfigHelper.embeddingsTmpDir).map(new Path(_)).getOrElse(
        spark.hadoopConfiguration.get("hadoop.tmp.dir")
      )
    }
    val clusterFilePath = Path.mergePaths(new Path(fileSystem.getUri.toString + clusterTmpLocation), new Path(clusterFileName))

    // 1 and 2.  Copy to local and Index Word Embeddings
    indexEmbeddings(sourceEmbeddingsPath, tmpLocalDestination.toString, format, spark)

    if (destinationScheme == "file") {
      new File(tmpLocalDestination.toString).renameTo(new File(EmbeddingsHelper.getLocalEmbeddingsPath(clusterFileName)))
    } else {
      // 2. Copy WordEmbeddings to cluster
      copyIndexToCluster(tmpLocalDestination.toString, clusterFilePath, spark)
      FileHelper.delete(tmpLocalDestination.toString)
    }

    // 3. Create Spark Embeddings
    new ClusterWordEmbeddings(clusterFileName, dim, caseSensitive)
  }
}
