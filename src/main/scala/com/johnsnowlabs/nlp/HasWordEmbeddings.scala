package com.johnsnowlabs.nlp

import java.io.File
import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.embeddings.{EmbeddingsHelper, SparkWordEmbeddings, WordEmbeddings, WordEmbeddingsFormat}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkContext, SparkFiles}


/**
  * Base class for models that uses Word Embeddings.
  * This implementation is based on RocksDB so it has a compact RAM usage
  *
  * Corresponding Approach have to implement AnnotatorWithWordEmbeddings
   */

trait HasWordEmbeddings extends AutoCloseable with ParamsAndFeaturesWritable {

  val nDims = new IntParam(this, "nDims", "Number of embedding dimensions")
  val indexPath = new Param[String](this, "indexPath", "File that stores Index")
  val includeEmbeddings = new BooleanParam(this, "includeEmbeddings", "whether to include embeddings when saving annotator")
  val useNormalizedTokensForEmbeddings = new BooleanParam(this, "useNormalizedTokensForEmbeddings", "whether to use embeddings of normalized tokens (if not already normalized)")

  def setDims(nDims: Int): this.type = set(this.nDims, nDims)
  def setIndexPath(path: String): this.type = set(this.indexPath, path)
  def setIncludeEmbeddings(value: Boolean): this.type = set(this.includeEmbeddings, value)
  def setUseNormalizedTokensForEmbeddings(value: Boolean): this.type = set(this.useNormalizedTokensForEmbeddings, value)

  def setEmbeddings(path: String, spark: SparkSession): this.type = {
    if (sparkEmbeddings == null)
      sparkEmbeddings = new SparkWordEmbeddings($(indexPath), $(nDims), $(useNormalizedTokensForEmbeddings))
    else
      throw new UnsupportedOperationException("Trying to override a already set embeddings")
    this
  }

  def setEmbeddings(embeddings: SparkWordEmbeddings): Unit = {
    if (sparkEmbeddings == null)
      sparkEmbeddings = embeddings
    else
      throw new UnsupportedOperationException("Trying to override a already set embeddings")
  }

  def saveEmbeddings(path: String, spark: SparkSession): Unit = {
    serializeEmbeddings(path, spark.sparkContext)
  }

  setDefault(useNormalizedTokensForEmbeddings, true)

  @transient
  private[nlp] var sparkEmbeddings: SparkWordEmbeddings = _

  def embeddings: Option[WordEmbeddings] = get(indexPath).map { path =>
    if (sparkEmbeddings == null)
      sparkEmbeddings = new SparkWordEmbeddings(path, $(nDims), $(useNormalizedTokensForEmbeddings))

    sparkEmbeddings.wordEmbeddings
  }

  override def close(): Unit = {
    if (embeddings.nonEmpty)
      embeddings.get.close()
  }

  def moveFolderFiles(folderSrc: String, folderDst: String): Unit = {
    for (file <- new File(folderSrc).list()) {
      Files.move(Paths.get(folderSrc, file), Paths.get(folderDst, file))
    }

    Files.delete(Paths.get(folderSrc))
  }

  def deserializeEmbeddings(path: String, spark: SparkContext): Unit = {
    val embeddings = EmbeddingsHelper.loadEmbeddings(path, spark, $(useNormalizedTokensForEmbeddings))
    if (embeddings.isDefined) {
      setIndexPath(embeddings.get.clusterFilePath.toString)
      sparkEmbeddings = embeddings.get
    }
  }

  def serializeEmbeddings(path: String, spark: SparkContext): Unit = {
    if ($(includeEmbeddings) && isDefined(indexPath)) {
      EmbeddingsHelper.saveEmbeddings(path, spark, $(indexPath))
    }
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    serializeEmbeddings(path, spark.sparkContext)
  }
}