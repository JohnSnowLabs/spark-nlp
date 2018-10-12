package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param}

trait HasLazyEmbeddings extends AutoCloseable with ParamsAndFeaturesWritable {

  @transient
  protected var clusterEmbeddings: Option[SparkWordEmbeddings] = None

  val includeEmbeddings = new BooleanParam(this, "includeEmbeddings", "whether to include embeddings when saving annotator")
  val includedEmbeddingsRef = new Param[String](this, "includedEmbeddingsRef", "if sourceEmbeddingsPath was provided, name them with this ref. Otherwise, use embeddings by this ref")

  setDefault(includeEmbeddings, true)

  def setIncludeEmbeddings(value: Boolean): this.type = set(this.includeEmbeddings, value)
  def setIncludedEmbeddingsRef(value: String): this.type = set(this.includedEmbeddingsRef, value)

  val caseSensitiveEmbeddings = new BooleanParam(this, "caseSensitiveEmbeddings", "whether to ignore case in tokens for embeddings matching")
  val embeddingsDim = new IntParam(this, "nDims", "Number of embedding dimensions")

  setDefault(caseSensitiveEmbeddings, true)

  def setCaseSensitiveEmbeddings(value: Boolean): this.type = set(this.caseSensitiveEmbeddings, value)
  def setEmbeddingsDim(value: Int): this.type = set(this.embeddingsDim, value)

  def setEmbeddings(embeddings: SparkWordEmbeddings): Unit = {
    set(embeddingsDim, embeddings.dim)
    set(caseSensitiveEmbeddings, embeddings.caseSensitive)
    clusterEmbeddings = Some(embeddings)
  }

  def setEmbeddingsIfFNotSet(embeddings: SparkWordEmbeddings): Unit = {
    if (clusterEmbeddings.isEmpty) {
      setEmbeddings(embeddings)
    }
  }

  def getEmbeddings: SparkWordEmbeddings = {
    clusterEmbeddings.getOrElse(throw new NoSuchElementException(s"embeddings not set in $uid"))
  }

  override def close(): Unit = {
    if (clusterEmbeddings.nonEmpty)
      clusterEmbeddings.get.wordEmbeddings.close()
  }

}
