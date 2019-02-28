package com.johnsnowlabs.nlp.embeddings

import java.io.File
import java.nio.file.{Files, Paths}

import org.apache.hadoop.fs.{FileSystem, Path}
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

  def moveFolderFiles(folderSrc: String, folderDst: String): Unit = {
    for (file <- new File(folderSrc).list()) {
      Files.move(Paths.get(folderSrc, file), Paths.get(folderDst, file))
    }

    Files.delete(Paths.get(folderSrc))
  }

  def deserializeEmbeddings(path: String, spark: SparkSession): Unit = {
    val src = getEmbeddingsSerializedPath(path)


    EmbeddingsHelper.load(
      src.toUri.toString,
      spark,
      WordEmbeddingsFormat.SPARKNLP.toString,
      $(embeddingsDim),
      $(caseSensitiveEmbeddings),
      $(embeddingsRef)
    )
  }

  def serializeEmbeddings(path: String, spark: SparkSession): Unit = {
    val index = new Path(EmbeddingsHelper.getLocalEmbeddingsPath(getClusterEmbeddings.fileName))

    val uri = new java.net.URI(path)
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dst = getEmbeddingsSerializedPath(path)

    EmbeddingsHelper.save(fs, index, dst)
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    /** Param only useful for runtime execution */
    serializeEmbeddings(path, spark)
  }
}
