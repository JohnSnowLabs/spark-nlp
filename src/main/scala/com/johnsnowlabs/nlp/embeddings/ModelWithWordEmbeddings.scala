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

  def moveFolderFiles(folderSrc: String, folderDst: String): Unit = {
    for (file <- new File(folderSrc).list()) {
      Files.move(Paths.get(folderSrc, file), Paths.get(folderDst, file))
    }

    Files.delete(Paths.get(folderSrc))
  }

  def deserializeEmbeddings(path: String, spark: SparkSession): Unit = {
    val src = getEmbeddingsSerializedPath(path)

    if ($(includeEmbeddings)) {

      val clusterEmbeddings = EmbeddingsHelper.loadEmbeddings(
        src.toUri.toString,
        spark,
        WordEmbeddingsFormat.SPARKNLP.toString,
        $(embeddingsDim),
        $(caseSensitiveEmbeddings)
      )

      /** Set embeddings ref */
      EmbeddingsHelper.setEmbeddingsRef($(embeddingsRef), clusterEmbeddings)

    } else if (isSet(embeddingsRef)) {

      val clusterEmbeddings = EmbeddingsHelper
        .getEmbeddingsByRef($(embeddingsRef))
        .getOrElse(throw new NoSuchElementException(
          s"Embeddings for stage $uid not included and not found in embeddings cache by ref '${$(embeddingsRef)}'. " +
          s"Please load embeddings first using EmbeddingsHelper .loadEmbeddings() and .setEmbeddingsRef() by '${$(embeddingsRef)}'"
        ))
      setEmbeddingsDim(clusterEmbeddings.dim)
      setCaseSensitiveEmbeddings(clusterEmbeddings.caseSensitive)

    } else throw new IllegalArgumentException("Annotator requires embeddings. They're either not included or ref is not defined")

  }

  def serializeEmbeddings(path: String, spark: SparkSession): Unit = {
    if ($(includeEmbeddings)) {
      val index = new Path(SparkFiles.get(getClusterEmbeddings.clusterFilePath))

      val uri = new File(path).toURI
      val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
      val dst = getEmbeddingsSerializedPath(path)

      EmbeddingsHelper.saveEmbeddings(fs, index, dst)
    }
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    /** Param only useful for runtime execution */
    serializeEmbeddings(path, spark)
  }
}
