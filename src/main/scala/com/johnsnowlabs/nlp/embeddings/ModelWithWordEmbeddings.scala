package com.johnsnowlabs.nlp.embeddings

import java.io.File
import java.nio.file.{Files, Paths}

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkFiles
import org.apache.spark.sql.SparkSession

/**
  * Base class for models that uses Word Embeddings.
  * This implementation is based on RocksDB so it has a compact RAM usage
  *
  * Corresponding Approach have to implement AnnotatorWithWordEmbeddings
   */

trait ModelWithWordEmbeddings extends HasEmbeddings {

  private def getEmbeddingsSerializedPath(path: String): Path =
    Path.mergePaths(new Path(path), new Path("/embeddings"))

  private def updateAvailableEmbeddings: Unit = {
    /** clusterEmbeddings may become null when a different thread calls getEmbeddings. Clean up now. */
    val cleanEmbeddings: Option[SparkWordEmbeddings] = if (clusterEmbeddings == null) None else clusterEmbeddings
    val currentEmbeddings = cleanEmbeddings
      .orElse(get(includedEmbeddingsRef).flatMap(ref => EmbeddingsHelper.embeddingsCache.get(ref)))
      .orElse(get(includedEmbeddingsIndexPath).flatMap(path => EmbeddingsHelper.loadEmbeddings(path, $(embeddingsDim), $(caseSensitiveEmbeddings))))
      .getOrElse(throw new NoSuchElementException(
        s"Word embeddings missing. " +
          s"Not in ref cache ${get(includedEmbeddingsRef).getOrElse("")} " +
          s"or embeddings not included. Check includeEmbeddings or includedEmbeddingsRef params")
      )

    setEmbeddingsIfFNotSet(currentEmbeddings)
  }

  def getEmbeddings: SparkWordEmbeddings = {
    updateAvailableEmbeddings
    clusterEmbeddings.getOrElse(throw new NoSuchElementException(s"embeddings not set in $uid"))
  }

  def getWordEmbeddings: WordEmbeddings = {
    getEmbeddings.wordEmbeddings
  }

  def moveFolderFiles(folderSrc: String, folderDst: String): Unit = {
    for (file <- new File(folderSrc).list()) {
      Files.move(Paths.get(folderSrc, file), Paths.get(folderDst, file))
    }

    Files.delete(Paths.get(folderSrc))
  }

  def deserializeEmbeddings(path: String, spark: SparkSession): Unit = {
    val src = getEmbeddingsSerializedPath(path)
    val embeddings: Option[SparkWordEmbeddings] =
      get(includedEmbeddingsRef)
        .flatMap(EmbeddingsHelper.embeddingsCache.get)
        .orElse(EmbeddingsHelper.loadEmbeddings(
          src.toUri.toString,
          spark,
          WordEmbeddingsFormat.SPARKNLP,
          $(embeddingsDim),
          $(caseSensitiveEmbeddings)))

    embeddings.foreach(setEmbeddings)
  }

  def serializeEmbeddings(path: String, spark: SparkSession): Unit = {
    if ($(includeEmbeddings)) {
      val index = new Path(SparkFiles.get(getEmbeddings.clusterFilePath))

      val uri = new java.net.URI(path)
      val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
      val dst = getEmbeddingsSerializedPath(path)

      EmbeddingsHelper.saveEmbeddings(fs, index, dst)
    }
  }

  override def beforeWrite(): Unit = {
    clear(includedEmbeddingsIndexPath)
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    /** Param only useful for runtime execution */
    serializeEmbeddings(path, spark)
  }
}
