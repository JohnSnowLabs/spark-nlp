package com.johnsnowlabs.nlp

import java.io.{File, FileNotFoundException}
import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.embeddings._
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

  def embeddings: WordEmbeddings = {
    if (clusterEmbeddings.isEmpty) {
      if ($(includeEmbeddings) && isDefined(indexPath)) {
        clusterEmbeddings = Some(new SparkWordEmbeddings($(indexPath), $(embeddingsDim), $(caseSensitiveEmbeddings)))
        clusterEmbeddings.get.wordEmbeddings
      } else if (isDefined(includedEmbeddingsRef)) {
        clusterEmbeddings = Some(EmbeddingsHelper.embeddingsCache($(includedEmbeddingsRef)))
        clusterEmbeddings.get.wordEmbeddings
      } else throw new FileNotFoundException("Word embeddings missing. Not provided by source, index or instance")
    } else
      clusterEmbeddings.get.wordEmbeddings
  }

  def moveFolderFiles(folderSrc: String, folderDst: String): Unit = {
    for (file <- new File(folderSrc).list()) {
      Files.move(Paths.get(folderSrc, file), Paths.get(folderDst, file))
    }

    Files.delete(Paths.get(folderSrc))
  }

  def deserializeEmbeddings(path: String, spark: SparkSession): Unit = {
    val src = EmbeddingsHelper.getEmbeddingsSerializedPath(path)
    val embeddings: SparkWordEmbeddings = if ($(includeEmbeddings)) {
      EmbeddingsHelper.loadEmbeddings(
        src.toUri.toString,
        spark,
        WordEmbeddingsFormat.SPARKNLP,
        $(embeddingsDim),
        $(caseSensitiveEmbeddings))
    } else if (isDefined(includedEmbeddingsRef)) {
      EmbeddingsHelper.embeddingsCache.getOrElse($(includedEmbeddingsRef), throw new IndexOutOfBoundsException("EmbeddingsRef not found in cache"))
    } else throw new IllegalArgumentException("Embeddings not found. Either they were not included or embeddingsRef not provided")

    setEmbeddings(embeddings)
  }

  def serializeEmbeddings(path: String, spark: SparkSession): Unit = {
    if ($(includeEmbeddings) && isDefined(indexPath)) {
      EmbeddingsHelper.saveEmbeddings(path, spark, $(indexPath))
    }
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    serializeEmbeddings(path, spark)
  }
}