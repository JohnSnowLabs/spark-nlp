package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param}

trait HasEmbeddings extends AutoCloseable with ParamsAndFeaturesWritable {

  val includeEmbeddings = new BooleanParam(this, "includeEmbeddings", "whether to include embeddings when saving annotator")
  val embeddingsRef = new Param[String](this, "embeddingsRef", "if sourceEmbeddingsPath was provided, name them with this ref. Otherwise, use embeddings by this ref")
  val embeddingsDim = new IntParam(this, "embeddingsDim", "Number of embedding dimensions")
  val caseSensitiveEmbeddings = new BooleanParam(this, "caseSensitiveEmbeddings", "whether to ignore case in tokens for embeddings matching")

  setDefault(includeEmbeddings, true)
  setDefault(caseSensitiveEmbeddings, false)
  setDefault(embeddingsRef, this.uid)

  def setIncludeEmbeddings(value: Boolean): this.type = set(this.includeEmbeddings, value)
  def setEmbeddingsRef(value: String): this.type = set(this.embeddingsRef, value)
  def setEmbeddingsDim(value: Int): this.type = set(this.embeddingsDim, value)
  def setCaseSensitiveEmbeddings(value: Boolean): this.type = set(this.caseSensitiveEmbeddings, value)

  protected lazy val preloadedEmbeddings: ClusterWordEmbeddings =
    EmbeddingsHelper.load(EmbeddingsHelper.getClusterPath($(embeddingsRef)), $(embeddingsDim), $(caseSensitiveEmbeddings))

  def getClusterEmbeddings: ClusterWordEmbeddings = {
    preloadedEmbeddings
  }

  override def close(): Unit = {
    get(embeddingsRef)
      .map(_ => preloadedEmbeddings)
      .foreach(_.getLocalRetriever.close())
  }

}
