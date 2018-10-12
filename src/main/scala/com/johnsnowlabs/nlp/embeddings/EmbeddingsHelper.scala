package com.johnsnowlabs.nlp.embeddings

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkFiles

import scala.collection.mutable.{Map => MMap}

object EmbeddingsHelper {

  val embeddingsCache = MMap.empty[String, SparkWordEmbeddings]

  def loadEmbeddings(
                      path: String,
                      spark: SparkSession,
                      format: WordEmbeddingsFormat.Format,
                      nDims: Int,
                      caseSensitiveEmbeddings: Boolean,
                      placeInCache: Option[String] = None
                    ): Option[SparkWordEmbeddings] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val src = new Path(path)

    if (fs.exists(src)) {
      val someEmbeddings =
        SparkWordEmbeddings(spark.sparkContext, src.toUri.toString, nDims, caseSensitiveEmbeddings, format)
      placeInCache.foreach(embeddingsCache.update(_, someEmbeddings))
      Some(someEmbeddings)
    } else {
      None
    }
  }

  def loadEmbeddings(
                      path: String,
                      spark: SparkSession,
                      format: String,
                      nDims: Int,
                      caseSensitiveEmbeddings: Boolean,
                      placeInCache: String
                    ): Option[SparkWordEmbeddings] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val src = new Path(path)

    if (fs.exists(src)) {
      import WordEmbeddingsFormat._
      val someEmbeddings =
        SparkWordEmbeddings(spark.sparkContext, src.toUri.toString, nDims, caseSensitiveEmbeddings, str2frm(format))
      if (placeInCache.nonEmpty) embeddingsCache.update(placeInCache, someEmbeddings)
      Some(someEmbeddings)
    } else {
      None
    }
  }

  def getEmbeddingsFromAnnotator(annotator: ModelWithWordEmbeddings): SparkWordEmbeddings = {
    annotator.getEmbeddings
  }

  protected def saveEmbeddings(path: String, spark: SparkSession, indexPath: String): Unit = {
    val index = new Path(SparkFiles.get(indexPath))

    val uri = new java.net.URI(path)
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dst = new Path(path)

    saveEmbeddings(fs, index, dst)
  }

  def saveEmbeddings(path: String, embeddings: SparkWordEmbeddings, spark: SparkSession): Unit = {
    EmbeddingsHelper.saveEmbeddings(path, spark, embeddings.clusterFilePath.toString)
  }

  def saveEmbeddings(fs: FileSystem, index: Path, dst: Path): Unit = {
    fs.copyFromLocalFile(false, true, index, dst)
  }

  def clearCache: Unit = {
    embeddingsCache.clear()
  }

}
