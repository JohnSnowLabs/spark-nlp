package com.johnsnowlabs.nlp.embeddings

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkContext, SparkFiles}

object EmbeddingsHelper {

  def getEmbeddingsSerializedPath(path: String): Path = Path.mergePaths(new Path(path), new Path("/embeddings"))

  def loadEmbeddings(path: String, spark: SparkContext, useNormalizedTokensForEmbeddings: Boolean): Option[SparkWordEmbeddings] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.hadoopConfiguration)
    val src = getEmbeddingsSerializedPath(path)

    if (fs.exists(src)) {
      Some(SparkWordEmbeddings(spark, src.toUri.toString, 0, useNormalizedTokensForEmbeddings, WordEmbeddingsFormat.SPARKNLP))
    } else {
      None
    }
  }

  def saveEmbeddings(path: String, spark: SparkContext, indexPath: String): Unit = {
    val index = new Path(SparkFiles.get(indexPath))
    val uri = new java.net.URI(path)
    val fs = FileSystem.get(uri, spark.hadoopConfiguration)

    val dst = getEmbeddingsSerializedPath(path)
    fs.copyFromLocalFile(false, true, index, dst)
  }

  def saveEmbeddings(path: String, spark: SparkSession, embeddings: SparkWordEmbeddings): Unit = {
    saveEmbeddings(path, spark.sparkContext, embeddings.clusterFilePath.toString)
  }

}
