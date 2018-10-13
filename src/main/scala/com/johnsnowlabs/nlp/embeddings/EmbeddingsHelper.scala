package com.johnsnowlabs.nlp.embeddings

import java.io.{File, FileNotFoundException}

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkFiles

object EmbeddingsHelper {

  private val spark = ResourceHelper.spark

  private var embeddingsCache = spark.sparkContext.broadcast(Map.empty[String, ClusterWordEmbeddings])

  def setEmbeddingsRef(ref: String, embeddings: ClusterWordEmbeddings): Unit = {
    val current = embeddingsCache.value
    embeddingsCache.destroy()
    embeddingsCache = spark.sparkContext.broadcast(current ++ Map(ref -> embeddings))
  }

  def getEmbeddingsByRef(ref: String): Option[ClusterWordEmbeddings] = {
    embeddingsCache.value.get(ref)
  }

  def loadEmbeddings(
                      path: String,
                      spark: SparkSession,
                      format: String,
                      nDims: Int,
                      caseSensitiveEmbeddings: Boolean
                    ): ClusterWordEmbeddings = {
    val uri = new File(path).toURI
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val src = new Path(path)

    if (fs.exists(src)) {
      import WordEmbeddingsFormat._
      ClusterWordEmbeddings(
        spark.sparkContext,
        src.toUri.toString,
        nDims,
        caseSensitiveEmbeddings,
        str2frm(format)
      )
    } else {
      throw new FileNotFoundException(s"embeddings not found in $path")
    }
  }

  def loadEmbeddings(
                    path: String,
                    spark: SparkSession,
                    format: WordEmbeddingsFormat.Format,
                    nDims: Int,
                    caseSensitiveEmbeddings: Boolean): ClusterWordEmbeddings = {
    loadEmbeddings(
      path,
      spark,
      format.toString,
      nDims,
      caseSensitiveEmbeddings
    )
  }

  def loadEmbeddings(
                    indexPath: String,
                    nDims: Int,
                    caseSensitive: Boolean
                    ): ClusterWordEmbeddings = {
    new ClusterWordEmbeddings(indexPath, nDims, caseSensitive)
  }

  def getEmbeddingsFromAnnotator(annotator: ModelWithWordEmbeddings): ClusterWordEmbeddings = {
    annotator.getClusterEmbeddings
  }

  protected def saveEmbeddings(path: String, spark: SparkSession, indexPath: String): Unit = {
    val index = new Path(SparkFiles.get(indexPath))

    val uri = new File(path).toURI
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dst = new Path(path)

    saveEmbeddings(fs, index, dst)
  }

  def saveEmbeddings(path: String, embeddings: ClusterWordEmbeddings, spark: SparkSession): Unit = {
    EmbeddingsHelper.saveEmbeddings(path, spark, embeddings.clusterFilePath.toString)
  }

  def saveEmbeddings(fs: FileSystem, index: Path, dst: Path): Unit = {
    fs.copyFromLocalFile(false, true, index, dst)
  }

  def clearCache(): Unit = {
    embeddingsCache.destroy()
    embeddingsCache = spark.sparkContext.broadcast(Map.empty[String, ClusterWordEmbeddings])
  }

}
