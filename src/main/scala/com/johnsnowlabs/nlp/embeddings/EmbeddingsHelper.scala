package com.johnsnowlabs.nlp.embeddings

import java.io.FileNotFoundException

import com.johnsnowlabs.nlp.ModelWithWordEmbeddings
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkFiles

import scala.collection.mutable.{Map => MMap}

object EmbeddingsHelper {

  val embeddingsCache = MMap.empty[String, SparkWordEmbeddings]

  def getEmbeddingsSerializedPath(path: String): Path = Path.mergePaths(new Path(path), new Path("/embeddings"))

  def loadEmbeddings(
                      path: String,
                      spark: SparkSession,
                      format: WordEmbeddingsFormat.Format,
                      nDims: Int,
                      useNormalizedTokensForEmbeddings: Boolean): SparkWordEmbeddings = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val src = new Path(path)

    if (fs.exists(src)) {
      SparkWordEmbeddings(spark.sparkContext, src.toUri.toString, nDims, useNormalizedTokensForEmbeddings, format)
    } else {
      throw new FileNotFoundException(s"Could not load embeddings. ${src.toString} not found")
    }
  }

  def getEmbeddingsFromAnnotator(annotator: ModelWithWordEmbeddings): Option[SparkWordEmbeddings] = {
    annotator.getEmbeddings
  }

  def saveEmbeddings(path: String, spark: SparkSession, indexPath: String): Unit = {
    val index = new Path(SparkFiles.get(indexPath))
    val uri = new java.net.URI(path)
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)

    val dst = new Path(path)
    fs.copyFromLocalFile(false, true, index, dst)
  }

  def saveEmbeddings(path: String, spark: SparkSession, embeddings: SparkWordEmbeddings): Unit = {
    saveEmbeddings(path, spark, embeddings.clusterFilePath.toString)
  }

}
