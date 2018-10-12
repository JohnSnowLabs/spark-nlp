package com.johnsnowlabs.nlp

import java.io.File
import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.embeddings._
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkFiles
import org.apache.spark.ml.param.Param
import org.apache.spark.sql.SparkSession

/**
  * Base class for models that uses Word Embeddings.
  * This implementation is based on RocksDB so it has a compact RAM usage
  *
  * Corresponding Approach have to implement AnnotatorWithWordEmbeddings
   */

trait ModelWithWordEmbeddings extends HasLazyEmbeddings with ParamsAndFeaturesWritable {

  val indexPath = new Param[String](this, "indexPath", "File that stores Index")

  def setIndexPath(path: String): this.type = set(this.indexPath, path)

  private def getEmbeddingsSerializedPath(path: String): Path =
    Path.mergePaths(new Path(path), new Path("/embeddings"))

  private def updateAvailableEmbeddings: Unit = {
    val currentEmbeddings = clusterEmbeddings
      .orElse(get(includedEmbeddingsRef).flatMap(EmbeddingsHelper.embeddingsCache.get))
      .orElse(get(indexPath).map(new SparkWordEmbeddings(_, $(embeddingsDim), $(caseSensitiveEmbeddings))))
      .getOrElse(throw new NoSuchElementException(
        s"Word embeddings missing. " +
          s"Not in cache ${get(includedEmbeddingsRef).getOrElse("")} " +
          s"or index path not found ${get(indexPath).getOrElse("")}")
      )

    setEmbeddingsIfFNotSet(currentEmbeddings)
  }

  def embeddings: WordEmbeddings = {
    updateAvailableEmbeddings
    getEmbeddings.wordEmbeddings
  }

  def moveFolderFiles(folderSrc: String, folderDst: String): Unit = {
    for (file <- new File(folderSrc).list()) {
      Files.move(Paths.get(folderSrc, file), Paths.get(folderDst, file))
    }

    Files.delete(Paths.get(folderSrc))
  }

  def deserializeEmbeddings(path: String, spark: SparkSession): Unit = {
    val src = EmbeddingsHelper.getEmbeddingsSerializedPath(path)
    val embeddings: Option[SparkWordEmbeddings] =
      get(includedEmbeddingsRef)
        .flatMap(EmbeddingsHelper.embeddingsCache.get)
        .orElse(EmbeddingsHelper.loadEmbeddings(
          src.toUri.toString,
          spark,
          WordEmbeddingsFormat.SPARKNLP,
          $(embeddingsDim),
          $(caseSensitiveEmbeddings)))

    embeddings.foreach(e => {
      setIndexPath(e.clusterFilePath)
      setEmbeddings(e)
    })
  }

  def serializeEmbeddings(path: String, spark: SparkSession): Unit = {
    if ($(includeEmbeddings) && isDefined(indexPath)) {
      updateAvailableEmbeddings

      val index = new Path(SparkFiles.get(getEmbeddings.clusterFilePath))

      val uri = new java.net.URI(path)
      val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
      val dst = getEmbeddingsSerializedPath(path)

      EmbeddingsHelper.saveEmbeddings(fs, index, dst)
    }
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    serializeEmbeddings(path, spark)
  }
}