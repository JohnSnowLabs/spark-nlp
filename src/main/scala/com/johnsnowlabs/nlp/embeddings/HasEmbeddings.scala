package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param}

trait HasEmbeddings extends AutoCloseable with ParamsAndFeaturesWritable {

  val includeEmbeddings = new BooleanParam(this, "includeEmbeddings", "whether to include embeddings when saving annotator")
  val embeddingsRef = new Param[String](this, "embeddingsRef", "if sourceEmbeddingsPath was provided, name them with this ref. Otherwise, use embeddings by this ref")
  val embeddingsDim = new IntParam(this, "embeddingsDim", "Number of embedding dimensions")
  val caseSensitiveEmbeddings = new BooleanParam(this, "caseSensitiveEmbeddings", "whether to ignore case in tokens for embeddings matching")

  setDefault(includeEmbeddings, true)
  setDefault(embeddingsRef, this.uid)
  setDefault(caseSensitiveEmbeddings, false)

  def setIncludeEmbeddings(value: Boolean): this.type = set(this.includeEmbeddings, value)
  def setEmbeddingsRef(value: String): this.type = set(this.embeddingsRef, value)
  def setEmbeddingsDim(value: Int): this.type = set(this.embeddingsDim, value)
  def setCaseSensitiveEmbeddings(value: Boolean): this.type = set(this.caseSensitiveEmbeddings, value)

  def getClusterEmbeddings: ClusterWordEmbeddings = {
    get(embeddingsRef)
      .orElse(getDefault(embeddingsRef))
      .flatMap(ref => EmbeddingsHelper.getByRef(ref))
      .getOrElse(throw new NoSuchElementException(
        s"embeddings not set in $uid for ref ${get(embeddingsRef).orElse(getDefault(embeddingsRef)).getOrElse("-ref not provided-")}")
      )
  }

  override def close(): Unit = {
    get(embeddingsRef)
      .flatMap(EmbeddingsHelper.getByRef)
      .foreach(_.getOrCreateLocalRetriever.close())
  }

}
