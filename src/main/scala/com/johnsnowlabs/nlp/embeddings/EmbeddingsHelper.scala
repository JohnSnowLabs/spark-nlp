package com.johnsnowlabs.nlp.embeddings

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

  def getEmbeddingsFromAnnotator(annotator: ModelWithWordEmbeddings): SparkWordEmbeddings = {
    annotator.getEmbeddings
  }

  def saveEmbeddings(path: String, spark: SparkSession, indexPath: String): Unit = {
    val index = new Path(SparkFiles.get(indexPath))

    val uri = new java.net.URI(path)
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dst = new Path(path)

    saveEmbeddings(fs, index, dst)
  }

  def saveEmbeddings(path: String, spark: SparkSession, embeddings: SparkWordEmbeddings): Unit = {
    EmbeddingsHelper.saveEmbeddings(path, spark, embeddings.clusterFilePath.toString)
  }

  def saveEmbeddings(fs: FileSystem, index: Path, dst: Path): Unit = {
    fs.copyFromLocalFile(false, true, index, dst)
  }

  def clearCache: Unit = {
    embeddingsCache.clear()
  }

}
